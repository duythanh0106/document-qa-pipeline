"""
RAG Evaluation Scorer with LLM
Scores question-answer pairs using semantic evaluation (1-10 scale)
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DEFAULT_PROVIDER,
    OLLAMA_TIMEOUT,
    PROVIDER_CONFIGS,
    PROVIDER_KIND_MAP,
)
from providers.base import LLMProvider
from providers.factory import create_provider, normalize_provider_name

# Constants
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SLEEP = 0.3
ENV_FILE = ".env"

SYSTEM_PROMPT = """
You are a strict semantic evaluator.

Given:
- Question
- Chunk (ground truth context)
- Answer

Your task:
Determine whether the Answer is semantically supported by the Chunk.

Classify into EXACTLY one label with appropriate score:

ENTAILED (Answer is supported by Chunk):
- PERFECT (10): Hoàn toàn chính xác, đầy đủ, diễn đạt tốt
- EXCELLENT (9): Chính xác, đầy đủ nhưng diễn đạt có thể tốt hơn
- GOOD (8): Chính xác nhưng thiếu vài chi tiết nhỏ không quan trọng
- ACCEPTABLE (7): Đúng ý chính nhưng thiếu một số thông tin phụ

NOT_SUPPORTED (Chunk lacks information):
- PARTIALLY_SUPPORTED (6): Một phần đúng, một phần không có trong chunk
- UNCLEAR (5): Không đủ thông tin để xác định
- MOSTLY_UNSUPPORTED (4): Phần lớn không được hỗ trợ bởi chunk

CONTRADICTED (Answer conflicts with Chunk):
- MINOR_ERROR (3): Có lỗi nhỏ hoặc thiếu sót đáng kể
- MAJOR_ERROR (2): Sai nghiêm trọng hoặc ngược lại với chunk
- COMPLETELY_WRONG (1): Hoàn toàn sai, bịa đặt thông tin

Rules:
- Do NOT reward verbosity.
- If the answer says "không có thông tin" while the chunk has data → CONTRADICTED (1-3).
- Implicit or paraphrased matches count as ENTAILED.
- Do NOT use external knowledge.
- Consider completeness, accuracy, and clarity when scoring.

Output JSON only in the following format:
{
  "label": "ENTAILED | NOT_SUPPORTED | CONTRADICTED",
  "score": number from 1 to 10,
  "confidence": number between 0 and 1,
  "reason": "short explanation in Vietnamese"
}
"""


@dataclass(frozen=True)
class EvaluatorConfig:
    """Runtime configuration for the evaluator."""

    input_path: str
    output_path: str
    provider: str
    temperature: float
    sleep_time: float
    ollama_model: str
    ollama_url: str
    base_url: Optional[str] = None
    model: Optional[str] = None


def build_user_prompt(item: Dict) -> str:
    """Build evaluation prompt from QA item."""
    return f"""
Question:
{item.get("question")}

Chunk:
{item.get("chunk")}

Answer:
{item.get("answer")}
""".strip()


def build_messages(provider_name: str, prompt: str) -> List[Dict[str, str]]:
    """Build message array based on provider type."""
    if provider_name == "anthropic":
        return [{"role": "user", "content": prompt}]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def call_llm(provider: LLMProvider, provider_name: str, item: Dict, temperature: float) -> Dict:
    """Call LLM for semantic evaluation."""
    prompt = build_user_prompt(item)
    messages = build_messages(provider_name, prompt)

    response_text = provider.chat(
        messages,
        temperature=temperature,
        max_tokens=2000,
    )

    if not response_text:
        raise ValueError("Empty response from LLM")

    # Parse JSON response
    try:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")


def map_score_to_evaluate(score: int) -> str:
    """Map numeric score to evaluation category."""
    if score >= 7:
        return "correct"
    elif score >= 4:
        return "unclear"
    else:
        return "incorrect"


def process_item(
    provider: LLMProvider,
    provider_name: str,
    config: EvaluatorConfig,
    item: Dict,
    idx: int,
) -> None:
    """Process a single QA item."""
    try:
        nli = call_llm(provider, provider_name, item, config.temperature)

        score = nli.get("score", 0)
        item["evaluate"] = map_score_to_evaluate(score)
        item["score"] = score
        item["check"] = nli.get("reason", "")

        label = nli.get("label", "UNKNOWN")
        confidence = nli.get("confidence", 0)
        print(f"[{idx}] {label} (score: {score}, confidence: {confidence:.2f})")

    except Exception as e:
        item["evaluate"] = "error"
        item["score"] = 0
        item["check"] = str(e)
        print(f"[{idx}] ERROR → {e}")


def run_evaluation(config: EvaluatorConfig, provider: LLMProvider, provider_name: str) -> None:
    """Main evaluation loop."""
    input_path = Path(config.input_path)
    
    # Auto-generate output filename
    output_path = input_path.with_name(f"{input_path.stem}_semantic_scored.json")

    # Load data
    data = json.loads(input_path.read_text(encoding="utf-8"))
    total = len(data)

    print(f"Loaded: {input_path.name}")
    print(f"Total items: {total}")
    print(f"Provider: {provider_name.upper()}")
    if config.base_url:
        print(f"Custom URL: {config.base_url}")
    if config.model:
        print(f"Model: {config.model}")
    print()

    # Process each item
    for idx, item in enumerate(data, start=1):
        process_item(provider, provider_name, config, item, idx)
        if idx < total:
            time.sleep(config.sleep_time)

    # Save results
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'='*80}")
    print_statistics(data)
    print(f"\n Saved to: {output_path}")


def print_statistics(data: List[Dict]) -> None:
    """Print evaluation statistics."""
    evaluate_counts = {"correct": 0, "incorrect": 0, "unclear": 0, "error": 0}
    score_sum = 0
    score_count = 0
    label_counts = {"ENTAILED": 0, "NOT_SUPPORTED": 0, "CONTRADICTED": 0, "UNKNOWN": 0}

    for item in data:
        evaluate = item.get("evaluate", "error")
        evaluate_counts[evaluate] = evaluate_counts.get(evaluate, 0) + 1

        score = item.get("score", 0)
        if score > 0:
            score_sum += score
            score_count += 1

    total = len(data)
    avg_score = score_sum / score_count if score_count > 0 else 0

    print("Evaluation Summary:")
    print(f"   Total      : {total}")
    print(f"   Correct    : {evaluate_counts['correct']:3d} ({evaluate_counts['correct']/total*100:5.1f}%)")
    print(f"   Unclear    : {evaluate_counts['unclear']:3d} ({evaluate_counts['unclear']/total*100:5.1f}%)")
    print(f"   Incorrect  : {evaluate_counts['incorrect']:3d} ({evaluate_counts['incorrect']/total*100:5.1f}%)")
    print(f"   Error      : {evaluate_counts['error']:3d} ({evaluate_counts['error']/total*100:5.1f}%)")
    print(f"   Avg Score  : {avg_score:.2f}/10")


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


def resolve_api_key(provider_name: str, cli_key: Optional[str]) -> Optional[str]:
    """Resolve API key from CLI or environment."""
    if cli_key:
        return cli_key
    env_key = f"{provider_name.upper()}_API_KEY"
    return os.getenv(env_key)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    provider_choices = sorted(PROVIDER_CONFIGS.keys()) + ["ollama"]

    parser = argparse.ArgumentParser(
        description="RAG Evaluation Scorer - Semantic evaluation with 1-10 scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with DeepSeek
  python evaluate.py --input qa_data.json --api-key YOUR_KEY
  
  # With OpenAI
  python evaluate.py --input qa_data.json --provider openai --api-key YOUR_KEY
  
  # With Anthropic Claude
  python evaluate.py --input qa_data.json --provider anthropic --api-key YOUR_KEY
  
  # With custom base URL (OpenAI-compatible gateway)
  python evaluate.py --input qa_data.json --provider openai --api-key YOUR_KEY \\
      --base-url https://mygateway.ubbox.service
  
  # With custom model
  python evaluate.py --input qa_data.json --provider deepseek --api-key YOUR_KEY \\
      --model deepseek-chat
  
  # With Ollama (local)
  python evaluate.py --input qa_data.json --provider ollama \\
      --ollama-model qwen2.5:7b
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file with QA pairs",
    )
    parser.add_argument(
        "--provider",
        choices=provider_choices,
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        help="Model name to use (overrides default for provider)",
    )
    parser.add_argument(
        "--api-key",
        help="LLM API key (or set {PROVIDER}_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        help="Custom base URL for OpenAI-compatible APIs",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:7b",
        help="Ollama model name (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for LLM (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Sleep time between requests in seconds (default: {DEFAULT_SLEEP})",
    )

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load environment variables
    load_env_file()
    env_ollama_url = os.getenv("OLLAMA_URL")
    env_ollama_model = os.getenv("OLLAMA_MODEL")

    # Normalize provider name
    provider_name = normalize_provider_name(args.provider)

    # Build configuration
    config = EvaluatorConfig(
        input_path=args.input,
        output_path="",  # Auto-generated
        provider=provider_name,
        temperature=args.temperature,
        sleep_time=args.sleep,
        ollama_model=env_ollama_model or args.ollama_model,
        ollama_url=env_ollama_url or args.ollama_url,
        base_url=args.base_url,
        model=args.model,
    )

    # Resolve API key if needed
    api_key: Optional[str] = None
    provider_kind = PROVIDER_KIND_MAP.get(config.provider)
    if provider_kind == "api":
        api_key = resolve_api_key(config.provider, args.api_key)
        if not api_key:
            provider_env = f"{config.provider.upper()}_API_KEY"
            print("API key required!")
            print(f"   Use: --api-key YOUR_KEY")
            print(f"   Or set env: export {provider_env}=YOUR_KEY")
            return

    # Create provider
    try:
        provider = create_provider(
            provider_name=config.provider,
            api_key=api_key,
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model,
            ollama_timeout=OLLAMA_TIMEOUT,
            base_url=config.base_url,
            model=config.model,
        )
    except ValueError as exc:
        print(f"{exc}")
        return

    # Run evaluation
    run_evaluation(config, provider, provider_name)


if __name__ == "__main__":
    main()
