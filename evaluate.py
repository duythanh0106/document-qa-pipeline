import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict


SYSTEM_PROMPT = """
You are a strict semantic evaluator.

Given:
- Question
- Chunk (ground truth context)
- Answer

Your task:
Determine whether the Answer is semantically supported by the Chunk.

Classify into EXACTLY one label:
- ENTAILED: The chunk clearly supports the answer (including paraphrases).
- CONTRADICTED: The chunk contradicts the answer or the answer denies existing information.
- NOT_SUPPORTED: The chunk does not contain enough information.

Rules:
- Do NOT reward verbosity.
- If the answer says "không có thông tin" while the chunk has data → CONTRADICTED.
- Implicit or paraphrased matches count as ENTAILED.
- Do NOT use external knowledge.

Output JSON only in the following format:
{
  "label": "ENTAILED | CONTRADICTED | NOT_SUPPORTED",
  "confidence": number between 0 and 1,
  "reason": "short explanation"
}
"""


def build_user_prompt(item: Dict) -> str:
    return f"""
Question:
{item.get("question")}

Chunk:
{item.get("chunk")}

Answer:
{item.get("answer")}
""".strip()


def call_llm(provider, api_key, model, item, temperature):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]

    if provider == "deepseek":
        url = "https://api.deepseek.com/chat/completions"
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def map_nli_to_fields(nli: Dict) -> Dict:
    label = nli["label"]

    if label == "ENTAILED":
        return {"evaluate": "correct", "score": 9, "check": nli["reason"]}
    if label == "CONTRADICTED":
        return {"evaluate": "incorrect", "score": 1, "check": nli["reason"]}
    return {"evaluate": "unclear", "score": 5, "check": nli["reason"]}


def run(args):
    input_path = Path(args.input)
    output_path = input_path.with_name(
        f"{input_path.stem}_semantic_scored.json"
    )

    data = json.loads(input_path.read_text(encoding="utf-8"))

    for idx, item in enumerate(data, start=1):
        try:
            nli = call_llm(
                provider=args.provider,
                api_key=args.api_key,
                model=args.model,
                item=item,
                temperature=args.temperature,
            )
            mapped = map_nli_to_fields(nli)

            item["evaluate"] = mapped["evaluate"]
            item["score"] = mapped["score"]
            item["check"] = mapped["check"]

            print(f"[{idx}] {nli['label']}")
            time.sleep(args.sleep)

        except Exception as e:
            item["evaluate"] = "error"
            item["score"] = 0
            item["check"] = str(e)
            print(f"[{idx}] ERROR → {e}")

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n✅ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser("Semantic QA Evaluator (LLM-based)")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--provider", required=True, choices=["deepseek", "openai"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.3)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
