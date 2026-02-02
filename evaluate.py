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
You are a semantic evaluator with contextual understanding.

Given:
- Question (provides context about what information is being asked)
- Chunk (ground truth context retrieved from document)
- Answer (response to be evaluated)

Your task:
Determine whether the Answer is semantically supported by the Chunk, WITH consideration of the Question context.

CRITICAL UNDERSTANDING:
- The Chunk was retrieved specifically for the Question (via anchor/RAG system)
- If Question asks about entity X and Chunk provides the answer, it's VALID for Answer to mention entity X
- Example: Q: "Chiết khấu của Vietlott?" + Chunk: "3%" → A: "Chiết khấu Vietlott là 3%" is CORRECT
- The retrieval system guarantees Chunk is from the correct context/section

Classify into EXACTLY one label with appropriate score:

ENTAILED (Answer is supported by Chunk):
- PERFECT (10): Hoàn toàn chính xác, đầy đủ, diễn đạt tốt
  * Tất cả thông tin trong answer có trong chunk hoặc được suy ra hợp lý từ question context
  * Diễn đạt rõ ràng, mạch lạc
  
- EXCELLENT (9): Chính xác, đầy đủ nhưng diễn đạt có thể tốt hơn
  * Thông tin đúng 100%
  * Answer nhắc lại entity từ question (VD: "Vietlott") khi chunk cung cấp giá trị - điều này là HỢP LỆ
  * Có thể cải thiện cách diễn đạt hoặc cấu trúc câu
  
- GOOD (8): Chính xác với suy luận hợp lý từ ngữ cảnh
  * Thông tin cốt lõi (số liệu, thời gian, tên riêng) chính xác 100%
  * Có thêm chi tiết được suy luận hợp lý từ chunk (VD: "rà soát" → "rà soát và điều chỉnh")
  * Answer kết hợp thông tin từ question và chunk một cách hợp lý
  * Phần bổ sung KHÔNG mâu thuẫn với chunk
  * Thiếu vài chi tiết nhỏ KHÔNG quan trọng
  
- ACCEPTABLE (7): Đúng ý chính nhưng thiếu thông tin phụ
  * Trả lời đúng câu hỏi chính
  * Thiếu một số chi tiết quan trọng có trong chunk
  * Không có thông tin sai lệch

NOT_SUPPORTED (Chunk lacks information):
- PARTIALLY_SUPPORTED (6): Một phần đúng, một phần không có bằng chứng
  * Một phần thông tin có trong chunk
  * Một phần thông tin KHÔNG THỂ suy luận được từ chunk VÀ không phải từ question context
  * Không có mâu thuẫn rõ ràng
  
- UNCLEAR (5): Không đủ thông tin để xác định
  * Chunk quá mơ hồ hoặc không liên quan
  * Không thể kết luận đúng hay sai
  
- MOSTLY_UNSUPPORTED (4): Phần lớn không được hỗ trợ
  * Chỉ có ít thông tin trùng khớp
  * Phần lớn nội dung không có trong chunk

CONTRADICTED (Answer conflicts with Chunk):
- MINOR_ERROR (3): Có lỗi nhỏ hoặc thiếu sót đáng kể
  * Sai một vài chi tiết không quá quan trọng
  * Hoặc: Chunk có thông tin (hoặc link/reference) nhưng answer nói "không có thông tin"
  
- MAJOR_ERROR (2): Sai nghiêm trọng hoặc ngược lại với chunk
  * Sai thông tin cốt lõi (số liệu, thời gian, địa điểm)
  * Kết luận trái ngược với chunk
  
- COMPLETELY_WRONG (1): Hoàn toàn sai, bịa đặt thông tin
  * Toàn bộ thông tin không có trong chunk
  * Hoặc 100% mâu thuẫn với chunk

Evaluation Rules:
1. **Question Context Matters**: Question cung cấp context về entity/topic. Answer có thể nhắc lại entity từ question nếu chunk cung cấp giá trị đúng.
   - Q: "X của Vietlott?" + Chunk: "X là 3%" → A: "X của Vietlott là 3%"  VALID
   - Q: "Ai làm Y?" + Chunk: "KAM/BD làm Y" → A: "KAM/BD làm Y"  VALID

2. **Core Information Priority**: Thông tin cốt lõi (số liệu, thời gian, tên riêng, địa điểm) trong chunk phải chính xác 100%

3. **Reasonable Inference**: Chấp nhận suy luận hợp lý từ ngữ cảnh (VD: "rà soát thu nhập" có thể ngụ ý "rà soát và điều chỉnh thu nhập")

4. **Context-Based**: Trong môi trường doanh nghiệp, chấp nhận các quy trình ngầm định (VD: "phê duyệt" thường đi kèm "ký duyệt")

5. **No External Knowledge**: KHÔNG sử dụng kiến thức bên ngoài chunk

6. **Link/Reference Check**: Nếu chunk chứa link hoặc reference:
   - Answer nói "không có thông tin" → CONTRADICTED (2-3)
   - Answer đúng khi: hướng dẫn user đến link/reference

7. **No Verbosity Reward**: Không thưởng điểm cho câu trả lời dài dòng

8. **Paraphrasing**: Diễn đạt khác nhau của cùng một ý vẫn được chấp nhận

9. **Time/Location Specificity**: Thông tin về thời điểm cụ thể (VD: "sau khi giao") phải có bằng chứng rõ ràng trong chunk, KHÔNG được suy luận tùy tiện

Examples:

Example 1 - Entity from Question (VALID):
Q: "Chiết khấu của Vietlott cho Cash voucher?"
Chunk: "CHIẾT KHẤU: 3% tổng doanh thu THÁNG"
A: "Chiết khấu của Vietlott cho Cash voucher là 3% tổng doanh thu tháng"
→ Score 9-10 (PERFECT/EXCELLENT): Answer nhắc lại entity từ question, giá trị 3% đúng với chunk 

Example 2 - Reasonable Inference (VALID):
Q: "Công ty rà soát thu nhập vào tháng nào?"
Chunk: "Công ty rà soát thu nhập vào tháng 04 hàng năm."
A: "Công ty rà soát và điều chỉnh thu nhập vào tháng 04"
→ Score 8 (GOOD): Thông tin cốt lõi đúng, "điều chỉnh" là suy luận hợp lý 

Example 3 - Invalid Time Addition (INVALID):
Q: "Ai thông báo OP kích hoạt đơn hàng?"
Chunk: "KAM/BD thông báo OP kích hoạt đơn hàng"
A: "KAM/BD thông báo OP kích hoạt đơn hàng sau khi giao"
→ Score 6 (PARTIALLY_SUPPORTED): "Sau khi giao" là thông tin thời điểm cụ thể không có trong chunk

Example 4 - Link Reference (CONTRADICTED):
Q: "Chiết khấu của IOMEDIA?"
Chunk: "CHIẾT KHẤU: Chi tiết - https://docs.google.com/..."
A: "Không có thông tin chiết khấu trong ngữ cảnh"
→ Score 2-3 (CONTRADICTED): Chunk có link chứa thông tin, answer phủ nhận là SAI 

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
