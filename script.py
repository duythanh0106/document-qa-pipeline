"""
RAG Evaluation Dataset Generator with LLM
Generates Vietnamese questions based on 4W1H framework (What, Why, When, How)
"""
import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DEFAULT_NUM_QUESTIONS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUTPUT,
    DEFAULT_PROVIDER,
    MAX_RETRY,
    MAX_TOKENS,
    OLLAMA_TIMEOUT,
    PROVIDER_CONFIGS,
    PROVIDER_KIND_MAP,
    TEMPERATURE,
)
from prompts import SYSTEM_MESSAGE, build_user_prompt
from providers.base import LLMProvider
from providers.factory import create_provider, normalize_provider_name

JSON_LIST_PATTERN = re.compile(r"\[.*\]", re.DOTALL)
PREVIEW_COUNT = 5
PREVIEW_CHUNK_LEN = 200
ENV_FILE = ".env"


@dataclass
class QAPair:
    """Question-Answer pair with context."""

    question: str
    file: str
    chunk: str
    answer: str = ""
    evaluate: str = ""
    score: str = ""
    check: str = ""


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the generator."""

    input_path: str
    output_path: str
    num_questions: int
    provider: str
    preview: bool
    ollama_model: str
    ollama_url: str


def parse_json_list(response_text: str) -> List[Dict]:
    """Extract and parse the first JSON list from a response."""

    json_match = JSON_LIST_PATTERN.search(response_text)
    if not json_match:
        return []

    raw_json = json_match.group(0)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        repaired = repair_invalid_json_escapes(raw_json)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            print(f"⚠️  Invalid JSON returned by LLM: {exc}")
            return []


def repair_invalid_json_escapes(raw_json: str) -> str:
    """Best-effort fix for invalid backslash escapes in JSON strings."""

    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    chars: List[str] = []
    idx = 0

    while idx < len(raw_json):
        ch = raw_json[idx]
        if ch != "\\":
            chars.append(ch)
            idx += 1
            continue

        if idx + 1 >= len(raw_json):
            chars.append("\\\\")
            idx += 1
            continue

        nxt = raw_json[idx + 1]
        if nxt in valid_escapes:
            chars.append(ch)
            chars.append(nxt)
            idx += 2
            continue

        if nxt == "u" and idx + 5 < len(raw_json):
            hex_part = raw_json[idx + 2 : idx + 6]
            if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                chars.append(ch)
                chars.append(nxt)
                chars.append(hex_part)
                idx += 6
                continue

        chars.append("\\\\")
        idx += 1

    return "".join(chars)


class QuestionGenerator:
    """Main generator - no chunking, full document analysis."""

    def __init__(
        self,
        document_path: str,
        provider: LLMProvider,
        provider_name: str,
    ):
        self.document_path = Path(document_path)
        self.provider = provider
        self.provider_name = provider_name

        self.document_content = self.document_path.read_text(
            encoding="utf-8", errors="ignore"
        )
        self.filename = self.document_path.name

        print(f"📖 Loaded: {self.filename}")
        print(f"   Size: {len(self.document_content):,} characters")
        print(f"   Provider: {self.provider_name.upper()}")

    def generate_questions(
        self, num_questions: int = DEFAULT_NUM_QUESTIONS
    ) -> List[QAPair]:
        target = num_questions
        attempt = 0
        valid_pairs: List[QAPair] = []

        print(f"\n🤖 Target questions: {target}")

        while len(valid_pairs) < target and attempt <= MAX_RETRY:
            attempt += 1
            print(f"\n🔁 LLM attempt {attempt}")

            prompt = self._build_prompt(target - len(valid_pairs))
            messages = self._build_messages(prompt)
            response_text = self.provider.chat(
                messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            generated = self._parse_questions(response_text)
            print(f"   Raw generated: {len(generated)}")

            self._extend_valid_pairs(generated, target, valid_pairs)
            print(f"   ✅ Valid so far: {len(valid_pairs)}")

            if len(valid_pairs) >= target:
                break

        if len(valid_pairs) < target:
            print(f"\n⚠️ Only generated {len(valid_pairs)}/{target} valid questions")

        self._print_statistics(valid_pairs)
        return valid_pairs

    def _build_prompt(self, remaining: int) -> str:
        return build_user_prompt(self.document_content, self.filename, remaining)

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        if self.provider_name == "anthropic":
            return [{"role": "user", "content": prompt}]

        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]

    def _parse_questions(self, response_text: str) -> List[Dict]:
        questions = parse_json_list(response_text)
        if questions:
            print(f"✅ Generated {len(questions)} questions from LLM")
            return questions

        if self.provider_name == "deepseek" and not self.provider.last_error:
            print("⚠️  No valid JSON found in response")

        return []

    def _extend_valid_pairs(
        self, generated: List[Dict], target: int, valid_pairs: List[QAPair]
    ) -> None:
        for item in generated:
            if len(valid_pairs) >= target:
                break

            answer_text = item.get("answer_location", "").strip()
            if not answer_text:
                continue

            chunk = extract_chunk_verbatim(self.document_content, answer_text)
            if not chunk:
                continue

            if any(q.question == item.get("question") for q in valid_pairs):
                continue

            valid_pairs.append(
                QAPair(
                    question=item.get("question", ""),
                    file=self.filename,
                    chunk=chunk,
                )
            )

    def _print_statistics(self, qa_pairs: List[QAPair]) -> None:
        if not qa_pairs:
            return

        type_counts: Dict[str, int] = {}
        for pair in qa_pairs:
            q_type = classify_question_type(pair.question)
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        print("\n📊 Question Distribution:")
        for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            ratio = count / len(qa_pairs) * 100
            print(f"   {q_type:<10}: {count:2d} ({ratio:.1f}%)")

        print(f"   TOTAL     : {len(qa_pairs)}")


def classify_question_type(question: str) -> str:
    q = question.lower().strip()

    if any(w in q for w in ["ai ", "ai là", "ai chịu", "ai có trách nhiệm"]):
        return "who"

    if any(w in q for w in ["ở đâu", "tại đâu", "thuộc đâu", "áp dụng ở"]):
        return "where"

    if any(
        w in q
        for w in ["khi nào", "thời gian", "thời điểm", "lúc nào", "deadline"]
    ):
        return "when"

    if any(w in q for w in ["tại sao", "vì sao", "lý do"]):
        return "why"

    if any(w in q for w in ["như thế nào", "làm thế nào", "làm sao", "cách"]):
        return "how"

    if any(
        w in q
        for w in ["điều kiện", "trường hợp", "khi nào thì", "nếu", "ngoại lệ"]
    ):
        return "condition"

    if any(
        w in q
        for w in ["gồm những", "bao gồm", "liệt kê", "các bước", "những gì"]
    ):
        return "list"

    if any(w in q for w in ["là gì", "gì", "nội dung"]):
        return "what"

    return "unknown"


def extract_chunk_verbatim(document_text: str, answer_text: str) -> str:
    """
    Trả về đoạn trích NGUYÊN VĂN nếu answer_text xuất hiện trong document_text.
    Nếu không match 100% → trả về chuỗi rỗng.
    """

    if not answer_text:
        return ""

    start = document_text.find(answer_text)
    if start == -1:
        return ""

    return document_text[start : start + len(answer_text)]


def build_arg_parser() -> argparse.ArgumentParser:
    provider_choices = sorted(PROVIDER_CONFIGS.keys()) + ["ollama", "api"]

    parser = argparse.ArgumentParser(
        description="RAG Evaluation Generator - Vietnamese questions based on 4W1H"
    )
    parser.add_argument(
        "--input", required=True, help="Input document (.md, .txt, etc.)"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=DEFAULT_NUM_QUESTIONS,
        help="Number of questions to generate (default: 20)",
    )
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument(
        "--provider",
        choices=provider_choices,
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview questions without saving",
    )
    return parser


def resolve_api_key(provider_name: str, cli_key: Optional[str]) -> Optional[str]:
    if cli_key:
        return cli_key
    env_key = f"{provider_name.upper()}_API_KEY"
    return os.getenv(env_key)


def preview_dataset(dataset: List[Dict]) -> bool:
    print("\n" + "=" * 80)
    print("PREVIEW (first 5 questions)")
    print("=" * 80)

    for i, item in enumerate(dataset[:PREVIEW_COUNT], 1):
        chunk_preview = item["chunk"][:PREVIEW_CHUNK_LEN]
        print(f"\n[{i}] Câu hỏi: {item['question']}")
        print(f"    File: {item['file']}")
        print(f"    Chunk: {chunk_preview}...")
        print("-" * 80)

    save = input("\n💾 Save to file? (y/n): ").strip().lower()
    return save == "y"


def load_env_file(env_path: str = ENV_FILE) -> None:
    """Load key=value pairs from a .env file into the process environment."""

    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    load_env_file()
    env_ollama_url = os.getenv("OLLAMA_URL")
    env_ollama_model = os.getenv("OLLAMA_MODEL")

    provider_name = normalize_provider_name(args.provider)
    app_config = AppConfig(
        input_path=args.input,
        output_path=args.output,
        num_questions=args.num_questions,
        provider=provider_name,
        preview=args.preview,
        ollama_model=env_ollama_model or args.ollama_model,
        ollama_url=env_ollama_url or args.ollama_url,
    )

    api_key: Optional[str] = None
    provider_kind = PROVIDER_KIND_MAP.get(app_config.provider)
    if provider_kind == "api":
        api_key = resolve_api_key(app_config.provider, args.api_key)
        if not api_key:
            provider_env = f"{app_config.provider.upper()}_API_KEY"
            print("❌ API key required!")
            print("   Use: --api-key YOUR_KEY")
            print(f"   Or set env: export {provider_env}=YOUR_KEY")
            return

    try:
        provider = create_provider(
            provider_name=app_config.provider,
            api_key=api_key,
            ollama_url=app_config.ollama_url,
            ollama_model=app_config.ollama_model,
            ollama_timeout=OLLAMA_TIMEOUT,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    generator = QuestionGenerator(
        app_config.input_path,
        provider=provider,
        provider_name=app_config.provider,
    )

    qa_pairs = generator.generate_questions(app_config.num_questions)
    if not qa_pairs:
        print("❌ No questions generated")
        return

    dataset = [asdict(pair) for pair in qa_pairs]

    if app_config.preview and not preview_dataset(dataset):
        print("Cancelled.")
        return

    with open(app_config.output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(dataset)} questions to: {app_config.output_path}")


if __name__ == "__main__":
    main()
