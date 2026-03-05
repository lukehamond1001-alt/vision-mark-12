"""Fetch and process mixed training data for hierarchical convolutional LLM.

Downloads and processes three sources:
  1. Simple English Wikipedia — plaintext articles (50% of corpus)
  2. OpenAssistant (oasst1)  — conversational Q&A (30% of corpus)
  3. Project Gutenberg       — public domain books (20% of corpus)

Output:
  data/wiki_plaintext.txt      — cleaned plain text from Wikipedia
  data/conversations.txt       — <INPUT>...</INPUT>response format conversations
  data/gutenberg_plaintext.txt — cleaned Gutenberg plain text

Usage:
  python scripts/fetch_data.py
  python scripts/fetch_data.py --output-dir data/ --wiki-chars 5000000
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Text cleaning (matches vm12/data.py)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize text to only use characters in our 99-char vocab."""
    replacements = {
        '\n': ' ', '\r': ' ', '\t': ' ',
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2014': '-', '\u2013': '-',
        '\u2026': '...', '\ufeff': '',
        '\u00e9': 'e', '\u00e8': 'e',
        '\u00e0': 'a', '\u00f1': 'n',
        '\u00fc': 'u', '\u00f6': 'o',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip any remaining non-ASCII
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    start_markers = ['*** START OF THE PROJECT GUTENBERG',
                     '*** START OF THIS PROJECT GUTENBERG']
    end_markers = ['*** END OF THE PROJECT GUTENBERG',
                   '*** END OF THIS PROJECT GUTENBERG']

    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            nl = text.find('\n')
            if nl != -1:
                text = text[nl + 1:]
            break

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


# ---------------------------------------------------------------------------
# Source 1: Simple English Wikipedia
# ---------------------------------------------------------------------------

def fetch_wikipedia(output_path: Path, max_chars: int = 5_000_000) -> int:
    """Download Simple English Wikipedia from Wikimedia dumps.

    Downloads the articles XML dump (bz2 compressed), parses incrementally
    to extract plain text from article bodies. Strips wikitext markup.
    """
    print("\n=== Fetching Simple English Wikipedia ===")

    import bz2
    import xml.etree.ElementTree as ET

    dump_url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"

    # Download the dump
    print(f"  Downloading dump (~200MB)...")
    tmp_bz2 = Path(tempfile.mktemp(suffix='.xml.bz2'))

    try:
        req = Request(dump_url, headers={'User-Agent': 'VM12-DataFetch/1.0'})
        with urlopen(req, timeout=300) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(tmp_bz2, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                        pct = downloaded / total_size * 100
                        print(f"    {downloaded / 1024 / 1024:.0f}MB / {total_size / 1024 / 1024:.0f}MB ({pct:.0f}%)")

        print(f"  Downloaded {downloaded / 1024 / 1024:.0f}MB, parsing articles...")

        # Parse bz2 XML incrementally
        # Auto-detect namespace from first element tag
        all_text = []
        total_chars = 0
        article_count = 0

        def _local_tag(tag: str) -> str:
            """Strip namespace prefix: {http://...}page -> page"""
            if '}' in tag:
                return tag.split('}', 1)[1]
            return tag

        with bz2.open(tmp_bz2, 'rt', encoding='utf-8', errors='ignore') as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                if total_chars >= max_chars:
                    break

                if _local_tag(elem.tag) == 'page':
                    # Skip non-article namespaces
                    ns_elem = next(
                        (c for c in elem if _local_tag(c.tag) == 'ns'), None
                    )
                    if ns_elem is not None and ns_elem.text != '0':
                        elem.clear()
                        continue

                    # Get article text (search descendants)
                    text_elem = None
                    for desc in elem.iter():
                        if _local_tag(desc.tag) == 'text' and desc.text:
                            text_elem = desc
                            break

                    if text_elem is not None and text_elem.text:
                        raw = text_elem.text

                        # Skip redirects
                        if raw.strip().lower().startswith('#redirect'):
                            elem.clear()
                            continue

                        # Strip wikitext markup
                        cleaned = _strip_wikitext(raw)
                        cleaned = clean_text(cleaned)

                        if len(cleaned) >= 100:
                            all_text.append(cleaned)
                            total_chars += len(cleaned)
                            article_count += 1

                            if article_count % 5000 == 0:
                                print(f"    {article_count} articles, {total_chars:,} chars")

                    elem.clear()

        text = ' '.join(all_text)[:max_chars]
        output_path.write_text(text, encoding='utf-8')
        print(f"  Wrote {len(text):,} chars from {article_count} articles to {output_path}")
        return len(text)

    except Exception as e:
        print(f"  Dump download/parse failed: {e}")
        print("  Falling back to Wikipedia API...")
        return _fetch_wikipedia_api(output_path, max_chars)

    finally:
        if tmp_bz2.exists():
            tmp_bz2.unlink()


def _strip_wikitext(text: str) -> str:
    """Remove common wikitext markup to extract plain text."""
    # Remove templates {{ }}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove categories/files [[Category:...]], [[File:...]]
    text = re.sub(r'\[\[(Category|File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # Convert wikilinks [[target|display]] → display, [[target]] → target
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
    # Remove external links [http://... display] → display
    text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove ref tags and their content
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    # Remove section headers == ... ==
    text = re.sub(r'={2,}[^=]+=+', ' ', text)
    # Remove bold/italic markers
    text = re.sub(r"'{2,}", '', text)
    # Remove tables {| ... |}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove remaining curly braces
    text = re.sub(r'[{}]', '', text)
    return text


def _fetch_wikipedia_api(output_path: Path, max_chars: int) -> int:
    """Fallback: fetch random Wikipedia articles via REST API."""
    print("  Using Wikipedia API fallback (slower)...")
    import json as _json
    base_url = "https://simple.wikipedia.org/api/rest_v1/page/random/summary"
    all_text = []
    total_chars = 0

    while total_chars < max_chars:
        try:
            req = Request(base_url, headers={'User-Agent': 'VM12-DataFetch/1.0'})
            with urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode('utf-8'))
                extract = data.get('extract', '')
                if len(extract) > 50:
                    cleaned = clean_text(extract)
                    all_text.append(cleaned)
                    total_chars += len(cleaned)
        except Exception:
            pass
        if total_chars % 50000 < 500:
            print(f"    {total_chars:,}/{max_chars:,} chars")

    text = ' '.join(all_text)[:max_chars]
    output_path.write_text(text, encoding='utf-8')
    print(f"  Wrote {len(text):,} chars to {output_path}")
    return len(text)


# ---------------------------------------------------------------------------
# Source 2: OpenAssistant (oasst1) conversations
# ---------------------------------------------------------------------------

def fetch_oasst1(output_path: Path, max_chars: int = 3_000_000) -> int:
    """Download and format OASST1 conversations from HuggingFace.

    Filters to English, extracts conversation threads, formats as
    <INPUT>question</INPUT>response pairs.
    """
    print("\n=== Fetching OpenAssistant (oasst1) ===")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        return 0

    print("  Loading dataset from HuggingFace...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    print(f"  Loaded {len(ds):,} messages")

    # Build message tree: parent_id → children
    messages = {}
    children = {}
    for msg in ds:
        msg_id = msg['message_id']
        messages[msg_id] = msg
        parent_id = msg.get('parent_id')
        if parent_id:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(msg_id)

    # Find root messages (prompter, no parent, English)
    roots = [
        msg_id for msg_id, msg in messages.items()
        if msg.get('parent_id') is None
        and msg.get('role') == 'prompter'
        and msg.get('lang') == 'en'
    ]
    print(f"  Found {len(roots):,} English root prompts")

    # Extract conversation threads (follow highest-ranked child at each step)
    conversations = []
    for root_id in roots:
        thread = _extract_thread(root_id, messages, children)
        if thread and len(thread) >= 2:
            conversations.append(thread)

    print(f"  Extracted {len(conversations):,} conversation threads")

    # Format as <INPUT>...</INPUT>response
    formatted_blocks = []
    total_chars = 0

    for thread in conversations:
        if total_chars >= max_chars:
            break

        block_parts = []
        for i in range(0, len(thread) - 1, 2):
            user_msg = thread[i]
            assistant_msg = thread[i + 1] if i + 1 < len(thread) else None
            if not assistant_msg:
                break

            user_text = clean_text(user_msg)
            assistant_text = clean_text(assistant_msg)

            if len(user_text) < 5 or len(assistant_text) < 5:
                continue
            # Cap very long messages for our char-level model
            user_text = user_text[:500]
            assistant_text = assistant_text[:1000]

            block_parts.append(f"<INPUT>{user_text}</INPUT>{assistant_text}")

        if block_parts:
            block = '\n'.join(block_parts)
            formatted_blocks.append(block)
            total_chars += len(block)

    output_text = '\n\n'.join(formatted_blocks)
    output_path.write_text(output_text, encoding='utf-8')
    print(f"  Wrote {len(output_text):,} chars to {output_path}")
    return len(output_text)


def _extract_thread(root_id: str, messages: dict, children: dict,
                    max_depth: int = 10) -> list[str]:
    """Extract a single conversation thread by following the best child."""
    thread = []
    current_id = root_id

    for _ in range(max_depth):
        msg = messages.get(current_id)
        if not msg:
            break

        text = msg.get('text', '').strip()
        if text:
            thread.append(text)

        # Follow the child with highest rank (or first child if no ranking)
        child_ids = children.get(current_id, [])
        if not child_ids:
            break

        # Prefer the highest-ranked child
        best_child = None
        best_rank = -1
        for cid in child_ids:
            child_msg = messages.get(cid, {})
            rank = child_msg.get('rank', 0) or 0
            if rank > best_rank:
                best_rank = rank
                best_child = cid

        if best_child is None:
            best_child = child_ids[0]

        current_id = best_child

    return thread


# ---------------------------------------------------------------------------
# Source 3: Project Gutenberg
# ---------------------------------------------------------------------------

GUTENBERG_URLS = [
    # Popular public domain works
    ("Pride and Prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("Moby Dick", "https://www.gutenberg.org/files/2701/2701-0.txt"),
    ("Frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
    ("Adventures of Sherlock Holmes", "https://www.gutenberg.org/files/1661/1661-0.txt"),
    ("Alice in Wonderland", "https://www.gutenberg.org/files/11/11-0.txt"),
    ("Great Expectations", "https://www.gutenberg.org/files/1400/1400-0.txt"),
    ("A Tale of Two Cities", "https://www.gutenberg.org/files/98/98-0.txt"),
    ("Dracula", "https://www.gutenberg.org/files/345/345-0.txt"),
    ("The War of the Worlds", "https://www.gutenberg.org/files/36/36-0.txt"),
    ("The Time Machine", "https://www.gutenberg.org/files/35/35-0.txt"),
    ("Treasure Island", "https://www.gutenberg.org/files/120/120-0.txt"),
    ("The Picture of Dorian Gray", "https://www.gutenberg.org/files/174/174-0.txt"),
    ("Little Women", "https://www.gutenberg.org/files/514/514-0.txt"),
    ("The Jungle Book", "https://www.gutenberg.org/files/236/236-0.txt"),
    ("The Scarlet Letter", "https://www.gutenberg.org/files/25344/25344-0.txt"),
    ("The Count of Monte Cristo", "https://www.gutenberg.org/files/1184/1184-0.txt"),
    ("Wuthering Heights", "https://www.gutenberg.org/files/768/768-0.txt"),
    ("Jane Eyre", "https://www.gutenberg.org/files/1260/1260-0.txt"),
    ("The Odyssey", "https://www.gutenberg.org/files/1727/1727-0.txt"),
    ("Don Quixote", "https://www.gutenberg.org/files/996/996-0.txt"),
]


def fetch_gutenberg(output_path: Path, max_chars: int = 2_000_000) -> int:
    """Download popular Gutenberg books as plain text."""
    print("\n=== Fetching Project Gutenberg ===")

    all_text = []
    total_chars = 0

    for title, url in GUTENBERG_URLS:
        if total_chars >= max_chars:
            break

        try:
            req = Request(url, headers={'User-Agent': 'VM12-DataFetch/1.0'})
            with urlopen(req, timeout=30) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')

            stripped = strip_gutenberg(raw)
            cleaned = clean_text(stripped)

            if len(cleaned) < 1000:
                print(f"  Skip {title} (too short after cleaning)")
                continue

            all_text.append(cleaned)
            total_chars += len(cleaned)
            print(f"  {title}: {len(cleaned):,} chars")

        except Exception as e:
            print(f"  Failed {title}: {e}")
            continue

    text = ' '.join(all_text)[:max_chars]
    output_path.write_text(text, encoding='utf-8')
    print(f"  Wrote {len(text):,} chars to {output_path}")
    return len(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch mixed training data for hierarchical conv LLM"
    )
    parser.add_argument("--output-dir", type=str, default="data/",
                        help="Output directory for processed text files")
    parser.add_argument("--wiki-chars", type=int, default=5_000_000,
                        help="Max characters from Wikipedia (default 5M)")
    parser.add_argument("--convo-chars", type=int, default=3_000_000,
                        help="Max characters from OASST1 (default 3M)")
    parser.add_argument("--gutenberg-chars", type=int, default=2_000_000,
                        help="Max characters from Gutenberg (default 2M)")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="Skip Wikipedia download")
    parser.add_argument("--skip-oasst", action="store_true",
                        help="Skip OASST1 download")
    parser.add_argument("--skip-gutenberg", action="store_true",
                        help="Skip Gutenberg download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = {}

    # Wikipedia
    wiki_path = output_dir / "wiki_plaintext.txt"
    if not args.skip_wiki:
        totals['wikipedia'] = fetch_wikipedia(wiki_path, args.wiki_chars)
    elif wiki_path.exists():
        totals['wikipedia'] = len(wiki_path.read_text())
        print(f"\nSkipping Wikipedia (exists: {totals['wikipedia']:,} chars)")

    # OASST1
    convo_path = output_dir / "conversations.txt"
    if not args.skip_oasst:
        oasst_chars = fetch_oasst1(convo_path, args.convo_chars)

        # Append existing Gutenberg conversations if they exist
        existing_convo = Path("gutenberg_convos.txt")
        if existing_convo.exists():
            existing_text = existing_convo.read_text()
            combined = convo_path.read_text() + "\n\n" + existing_text
            convo_path.write_text(combined)
            oasst_chars = len(combined)
            print(f"  Appended existing gutenberg_convos.txt ({len(existing_text):,} chars)")

        totals['conversations'] = oasst_chars
    elif convo_path.exists():
        totals['conversations'] = len(convo_path.read_text())
        print(f"\nSkipping OASST1 (exists: {totals['conversations']:,} chars)")

    # Gutenberg
    gutenberg_path = output_dir / "gutenberg_plaintext.txt"
    if not args.skip_gutenberg:
        totals['gutenberg'] = fetch_gutenberg(gutenberg_path, args.gutenberg_chars)
    elif gutenberg_path.exists():
        totals['gutenberg'] = len(gutenberg_path.read_text())
        print(f"\nSkipping Gutenberg (exists: {totals['gutenberg']:,} chars)")

    # Summary
    grand_total = sum(totals.values())
    print(f"\n{'='*50}")
    print(f"DATA SUMMARY")
    print(f"{'='*50}")
    for source, chars in totals.items():
        pct = chars / grand_total * 100 if grand_total > 0 else 0
        print(f"  {source:20s}: {chars:>10,} chars ({pct:.1f}%)")
    print(f"  {'TOTAL':20s}: {grand_total:>10,} chars")

    if grand_total < 5_000_000:
        print(f"\n  WARNING: Total corpus ({grand_total:,}) is below 5M char minimum.")
        print(f"  Consider increasing --wiki-chars or --gutenberg-chars.")
    else:
        print(f"\n  Corpus meets 10M+ target." if grand_total >= 10_000_000
              else f"\n  Corpus meets 5M minimum but below 10M target.")

    print(f"\nFiles saved to {output_dir}/")
    print(f"  Train with: python -m vm12.staged_trainer --text-path {output_dir}")


if __name__ == "__main__":
    main()
