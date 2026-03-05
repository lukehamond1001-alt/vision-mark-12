"""Generate synthetic conversation training data using Google Gemini API (free tier).

Produces <INPUT>...</INPUT> delimited text files for training Vision Mark 10.
Supports both single-turn Q&A and multi-turn conversations.

Rotates across multiple Gemini models to maximize free-tier throughput
(each model has its own 5 RPM limit).

Setup:
    1. Get a free API key at https://aistudio.google.com/app/apikey
    2. export GEMINI_API_KEY="your-key-here"

Usage:
    python3 scripts/generate_data.py --num-convos 500
    python3 scripts/generate_data.py --num-convos 1000 --multi-turn-ratio 0.5
"""

import argparse
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from google import genai

# ---------------------------------------------------------------------------
# Model rotation for free-tier rate limits (5 RPM per model)
# ---------------------------------------------------------------------------

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

class ModelRotator:
    """Rotate across models respecting per-model rate limits.

    Each free-tier model allows ~5 RPM. We track the last call time per model
    and pick whichever model has been idle longest. On 429 errors, we push
    that model's cooldown forward by 60s to avoid hammering it.
    """

    def __init__(self, client, rpm_per_model: int = 5):
        self.client = client
        self.rpm = rpm_per_model
        self.min_interval = 60.0 / rpm_per_model + 0.5
        self.ready_at: dict[str, float] = {}
        self.available: list[str] = []
        self._probe_models()

    def _probe_models(self):
        for model in MODELS:
            try:
                self.client.models.generate_content(
                    model=model, contents="Say hi",
                    config={"max_output_tokens": 5},
                )
                self.available.append(model)
                self.ready_at[model] = time.time() + self.min_interval
                print(f"  Model OK: {model}")
            except Exception:
                print(f"  Model skip: {model} (quota exhausted or unavailable)")
            time.sleep(1.5)

        if not self.available:
            raise RuntimeError("No Gemini models available. Try again later or check your API key.")
        print(f"  Using {len(self.available)} models (~{len(self.available) * self.rpm} effective RPM)\n")

    def _pick_model(self) -> tuple[str, float]:
        """Pick model with shortest wait. Returns (model, wait_seconds)."""
        now = time.time()
        best_model = self.available[0]
        best_wait = self.ready_at.get(best_model, 0) - now

        for model in self.available[1:]:
            wait = self.ready_at.get(model, 0) - now
            if wait < best_wait:
                best_wait = wait
                best_model = model

        return best_model, max(best_wait, 0)

    def call(self, contents: str, system: str, max_tokens: int = 400) -> str | None:
        """Make an API call with automatic model rotation and retry."""
        for attempt in range(len(self.available) + 2):
            model, wait = self._pick_model()
            if wait > 0:
                time.sleep(wait)

            self.ready_at[model] = time.time() + self.min_interval

            try:
                response = self.client.models.generate_content(
                    model=model, contents=contents,
                    config={
                        "system_instruction": system,
                        "temperature": 1.0,
                        "max_output_tokens": max_tokens,
                    },
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err:
                    self.ready_at[model] = time.time() + 65.0
                    continue
                print(f"  API error ({model}): {err[:80]}")
                return None
        return None


# ---------------------------------------------------------------------------
# Topics and prompts
# ---------------------------------------------------------------------------

SINGLE_TURN_TOPICS = [
    "a simple science fact",
    "a basic math question with numbers under 100",
    "a geography question about a specific country or city",
    "the definition of a common English word",
    "an interesting historical fact",
    "a question about a specific animal",
    "a cooking or recipe question",
    "a question about a popular sport",
    "a question about a musical instrument or genre",
    "a question about computers or the internet",
    "a common sense question",
    "a question about the human body or health",
    "a question about weather or seasons",
    "a question about a planet or star",
    "a question about a specific plant or flower",
    "a question a curious child might ask",
    "a polite greeting and a friendly response",
    "a short creative writing prompt",
    "a riddle with its answer",
    "a question about daily life or chores",
    "a question about colors, shapes, or patterns",
    "a question about time zones or calendars",
    "a question about a specific language or culture",
    "a math word problem",
    "a question about oceans, lakes, or rivers",
    "a question about insects or marine life",
    "a question about cars, trains, or planes",
    "a question about a famous book or author",
    "a question about feelings or emotions",
    "a question about recycling or the environment",
    "a vocabulary question (synonym, antonym, or usage)",
    "a question about a famous invention",
    "a question about a holiday or celebration",
    "a question about a job or career",
    "a question comparing two related things",
]

MULTI_TURN_SCENARIOS = [
    "a student asking a teacher about a science topic, with 2-3 follow-up questions",
    "someone learning to cook a simple recipe, asking step-by-step questions",
    "a curious child asking about an animal, then asking follow-ups about its habitat and diet",
    "someone asking for help with basic math, then asking about a related concept",
    "a person asking about a country, then its culture, then its food",
    "someone asking about a planet, then asking follow-up questions about space",
    "a person learning a new word, then asking for example sentences and synonyms",
    "someone asking about the weather, then asking how to prepare for it",
    "a person asking about a historical event, then asking about the people involved",
    "someone asking for a book recommendation, then asking about the plot and author",
    "a person asking how something works (like a phone or car), with follow-ups",
    "someone asking about a sport, then its rules, then famous players",
    "a person asking about healthy eating, then specific foods, then a meal idea",
    "someone asking about an instrument, then how to learn it, then famous musicians",
    "a person troubleshooting a simple problem step by step",
    "someone planning a trip and asking about destinations, packing, and activities",
    "a child asking why the sky is blue, then follow-up science questions",
    "someone learning about a holiday tradition with follow-up cultural questions",
    "a person asking about exercise, then specific workouts, then recovery tips",
    "someone asking about a career, then education needed, then daily tasks",
]

SINGLE_TURN_SYSTEM = """Generate exactly ONE question-answer pair. Rules:
- Question: under 80 characters, clear and specific
- Answer: 1-3 sentences, factual, concise, under 250 characters total
- Use ONLY plain ASCII characters (letters, digits, basic punctuation)
- No unicode, no emojis, no special symbols, no markdown formatting
- Format EXACTLY as shown below, nothing else before or after:

Q: <your question here>
A: <your answer here>"""

MULTI_TURN_SYSTEM = """Generate a short multi-turn conversation (2-4 exchanges). Rules:
- Each human message: under 80 characters
- Each assistant response: 1-3 sentences, under 250 characters
- Use ONLY plain ASCII characters (letters, digits, basic punctuation)
- No unicode, no emojis, no special symbols, no markdown formatting
- The conversation should flow naturally with follow-up questions
- Format EXACTLY as shown, nothing else before or after:

H: <human message 1>
A: <assistant response 1>
H: <human message 2>
A: <assistant response 2>"""

BATCH_SYSTEM_TEMPLATE = """Generate exactly {n} DIFFERENT question-answer pairs, numbered. Rules:
- Each question: under 80 characters, clear and specific
- Each answer: 1-3 sentences, factual, concise, under 250 characters
- Use ONLY plain ASCII characters (letters, digits, basic punctuation)
- No unicode, no emojis, no special symbols, no markdown formatting
- Format EXACTLY as:

1. Q: <question>
A: <answer>

2. Q: <question>
A: <answer>"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_single_turn(rotator: ModelRotator, topic: str) -> tuple[str, str] | None:
    prompt = f"Generate {topic}. Make it unique and interesting."
    text = rotator.call(prompt, SINGLE_TURN_SYSTEM, max_tokens=400)
    if not text:
        return None
    return _parse_single_turn(text)


def generate_multi_turn(rotator: ModelRotator, scenario: str) -> list[tuple[str, str]] | None:
    prompt = f"Generate a conversation about: {scenario}. Make it natural and informative."
    text = rotator.call(prompt, MULTI_TURN_SYSTEM, max_tokens=800)
    if not text:
        return None
    return _parse_multi_turn(text)


def generate_batch_single(rotator: ModelRotator, batch_size: int = 5) -> list[tuple[str, str]]:
    topics = random.sample(SINGLE_TURN_TOPICS, min(batch_size, len(SINGLE_TURN_TOPICS)))
    topic_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
    prompt = f"Generate {len(topics)} DIFFERENT question-answer pairs, one for each topic:\n{topic_list}"
    system = BATCH_SYSTEM_TEMPLATE.format(n=len(topics))

    text = rotator.call(prompt, system, max_tokens=2000)
    if not text:
        return []
    return _parse_batch_single(text)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_single_turn(text: str) -> tuple[str, str] | None:
    lines = text.strip().split('\n')
    question = None
    answer_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            question = line[2:].strip()
        elif line.startswith('A:'):
            answer_lines.append(line[2:].strip())
        elif answer_lines:
            answer_lines.append(line.strip())

    if not question or not answer_lines:
        return None

    answer = ' '.join(answer_lines).strip()
    if not _validate_pair(question, answer):
        return None
    return _sanitize(question), _sanitize(answer)


def _parse_multi_turn(text: str) -> list[tuple[str, str]] | None:
    lines = text.strip().split('\n')
    turns = []
    current_h = None
    current_a_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('H:'):
            if current_h and current_a_lines:
                turns.append((current_h, ' '.join(current_a_lines).strip()))
            current_h = line[2:].strip()
            current_a_lines = []
        elif line.startswith('A:'):
            current_a_lines.append(line[2:].strip())
        elif current_a_lines:
            current_a_lines.append(line.strip())

    if current_h and current_a_lines:
        turns.append((current_h, ' '.join(current_a_lines).strip()))

    valid_turns = []
    for h, a in turns:
        if _validate_pair(h, a):
            valid_turns.append((_sanitize(h), _sanitize(a)))

    return valid_turns if len(valid_turns) >= 2 else None


def _parse_batch_single(text: str) -> list[tuple[str, str]]:
    pairs = []
    cleaned = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    blocks = re.split(r'\n\s*\n|\n(?=Q:)', cleaned)
    for block in blocks:
        result = _parse_single_turn(block)
        if result:
            pairs.append(result)
    return pairs


def _validate_pair(question: str, answer: str) -> bool:
    if len(question) < 5 or len(answer) < 5:
        return False
    for bad in ('<INPUT>', '</INPUT>', '<START>', '<input>', '</input>'):
        if bad in question or bad in answer:
            return False
    return True


def _sanitize(text: str) -> str:
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_single_turn(q: str, a: str) -> str:
    return f"<INPUT>{q}</INPUT>{a}"


def format_multi_turn(turns: list[tuple[str, str]]) -> str:
    parts = []
    for h, a in turns:
        parts.append(f"<INPUT>{h}</INPUT>{a}")
    return "\n".join(parts)


def format_all(single_turns: list[tuple[str, str]],
               multi_turns: list[list[tuple[str, str]]]) -> str:
    blocks = []
    for q, a in single_turns:
        blocks.append(format_single_turn(q, a))
    for turns in multi_turns:
        blocks.append(format_multi_turn(turns))
    random.shuffle(blocks)
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data via Gemini API (free tier)")
    parser.add_argument("--num-convos", type=int, default=500,
                        help="Total conversations to generate")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.4,
                        help="Fraction of conversations that are multi-turn (0.0-1.0)")
    parser.add_argument("--output", type=str, default="data/conversations.txt",
                        help="Output file path")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N conversations")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Single-turn pairs per batch API call")
    args = parser.parse_args()

    import os
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable.")
        print("Get a free key at: https://aistudio.google.com/app/apikey")
        return

    client = genai.Client(api_key=api_key)

    print("Probing available models...")
    rotator = ModelRotator(client)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_multi = int(args.num_convos * args.multi_turn_ratio)
    num_single = args.num_convos - num_multi

    single_turns: list[tuple[str, str]] = []
    multi_turns: list[list[tuple[str, str]]] = []

    if output_path.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from vm7.data import _parse_conversations
        existing = output_path.read_text()
        convos = _parse_conversations(existing)
        for c in convos:
            if len(c) == 1:
                single_turns.append(c[0])
            else:
                multi_turns.append(c)
        print(f"Loaded {len(single_turns)} single + {len(multi_turns)} multi from existing file")

    failures = 0
    max_failures = args.num_convos * 3

    print(f"Target: {num_single} single-turn + {num_multi} multi-turn = {args.num_convos} conversations")
    print(f"Output: {output_path}\n")
    t0 = time.time()

    while len(single_turns) < num_single and failures < max_failures:
        batch = generate_batch_single(rotator, args.batch_size)
        if not batch:
            failures += 1
            continue

        for pair in batch:
            if len(single_turns) >= num_single:
                break
            single_turns.append(pair)

        done = len(single_turns) + len(multi_turns)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 0.1)
        eta = (args.num_convos - done) / max(rate, 0.01)
        print(f"  [{done}/{args.num_convos}] ({rate:.1f}/s, ~{eta/60:.1f}min left) "
              f"+{len(batch)} single-turn (total: {len(single_turns)})")

        if done % args.save_every < args.batch_size:
            output_path.write_text(format_all(single_turns, multi_turns))

    while len(multi_turns) < num_multi and failures < max_failures:
        scenario = random.choice(MULTI_TURN_SCENARIOS)
        result = generate_multi_turn(rotator, scenario)

        if result is None:
            failures += 1
            continue

        multi_turns.append(result)

        done = len(single_turns) + len(multi_turns)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 0.1)
        eta = (args.num_convos - done) / max(rate, 0.01)
        turns_str = " -> ".join(h[:30] for h, _ in result)
        print(f"  [{done}/{args.num_convos}] ({rate:.1f}/s, ~{eta/60:.1f}min left) "
              f"multi-turn ({len(result)} turns): {turns_str}")

        if done % args.save_every == 0:
            output_path.write_text(format_all(single_turns, multi_turns))

    output_path.write_text(format_all(single_turns, multi_turns))
    elapsed = time.time() - t0
    total = len(single_turns) + len(multi_turns)
    total_turns = len(single_turns) + sum(len(t) for t in multi_turns)
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Single-turn: {len(single_turns)}")
    print(f"  Multi-turn:  {len(multi_turns)} ({sum(len(t) for t in multi_turns)} total turns)")
    print(f"  Total training turns: {total_turns}")
    print(f"  Failures/skipped: {failures}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
