# RAG Evaluation Dataset Generator

Công cụ CLI sinh **bộ câu hỏi đánh giá RAG** (Vietnamese, theo khung 4W1H: What, Why, When, How) từ tài liệu đầu vào (`.md`, `.txt`, …), hỗ trợ nhiều LLM provider: **DeepSeek (default), OpenAI, Anthropic, Ollama**.

---

## 1. Yêu cầu

- Python ≥ 3.9
- Đã cài dependencies theo project
- Với Ollama: đã chạy Ollama local (`http://localhost:11434`)

---

## 2. Cách chạy cơ bản

### Default provider: **DeepSeek**

```bash
python script.py --input "path/to/doc.txt"
```

- Provider mặc định: `deepseek`
- API key lấy từ biến môi trường `DEEPSEEK_API_KEY` nếu không truyền `--api-key`

---

## 3. Truyền API key bằng flag `--api-key`

### DeepSeek (default)

```bash
python script.py \
  --input "path/to/doc.txt" \
  --api-key YOUR_DEEPSEEK_KEY
```

### OpenAI

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider openai \
  --api-key YOUR_OPENAI_KEY
```

### Anthropic

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider anthropic \
  --api-key YOUR_ANTHROPIC_KEY
```

---

## 4. Dùng API key qua biến môi trường (.env hoặc system env)

### DeepSeek (default)

```bash
python script.py --input "path/to/doc.txt"
```

Yêu cầu biến môi trường:

```env
DEEPSEEK_API_KEY=sk-xxxx
```

---

### OpenAI

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider openai
```

```env
OPENAI_API_KEY=sk-xxxx
```

---

### Anthropic

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider anthropic
```

```env
ANTHROPIC_API_KEY=sk-ant-xxxx
```

---

## 5. Sử dụng Custom Base URL (API Gateway/Proxy)

Bạn có thể chỉ định custom base URL cho các provider OpenAI-compatible (DeepSeek, OpenAI) thông qua flag `--base-url`.

### DeepSeek với custom gateway

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider deepseek \
  --api-key YOUR_KEY \
  --base-url https://api.custom-gateway.com
```

### OpenAI với custom gateway

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider openai \
  --api-key YOUR_KEY \
  --base-url https://mygateway.ubbox.service
```

### OpenAI-compatible endpoint khác

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider openai \
  --api-key YOUR_KEY \
  --base-url https://your-proxy.example.com/v1
```

### Kết hợp với các option khác

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider openai \
  --api-key YOUR_KEY \
  --base-url https://mygateway.ubbox.service \
  --num-questions 50 \
  --preview
```

**Lưu ý:**
- `--base-url` chỉ áp dụng cho **DeepSeek** và **OpenAI** providers
- **Anthropic** không hỗ trợ custom base URL (chỉ dùng official API)
- Đảm bảo gateway/proxy của bạn tương thích với OpenAI Chat Completions API format

---

## 6. Dùng Ollama (local LLM)

### Ollama – mặc định

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider ollama
```

Mặc định:
- URL: `http://localhost:11434`
- Model: lấy từ `OLLAMA_MODEL` hoặc config mặc định

---

### Chỉ định model và URL Ollama

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider ollama \
  --ollama-model "qwen2.5:7b" \
  --ollama-url "http://localhost:11434"
```

**Lưu ý:** Với Ollama, sử dụng `--ollama-url` thay vì `--base-url`

---

## 7. Alias provider `api`

Alias `api` sẽ map về **DeepSeek**:

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider api \
  --api-key YOUR_DEEPSEEK_KEY
```

---

## 8. Tuỳ chọn CLI thường dùng

```bash
--num-questions 30     # Số câu hỏi cần sinh
--output output.json   # File JSON đầu ra
--preview              # Xem trước 5 câu hỏi, hỏi có lưu hay không
--base-url URL         # Custom base URL cho API gateway/proxy
```

Ví dụ đầy đủ:

```bash
python script.py \
  --input "path/to/doc.txt" \
  --provider openai \
  --api-key YOUR_KEY \
  --base-url https://mygateway.ubbox.service \
  --num-questions 30 \
  --output my_eval.json \
  --preview
```

---

## 9. Bảng tổng hợp providers

| Provider | Default Base URL | Hỗ trợ `--base-url` | Environment Variable |
|----------|------------------|---------------------|----------------------|
| **deepseek** | `https://api.deepseek.com` | ✅ Yes | `DEEPSEEK_API_KEY` |
| **openai** | OpenAI official | ✅ Yes | `OPENAI_API_KEY` |
| **anthropic** | Anthropic official | ❌ No | `ANTHROPIC_API_KEY` |
| **ollama** | `http://localhost:11434` | ✅ Yes (dùng `--ollama-url`) | `OLLAMA_MODEL`, `OLLAMA_URL` |

---

## 10. Output

Kết quả được lưu dưới dạng JSON:

```json
[
  {
    "question": "...",
    "file": "doc.txt",
    "chunk": "...",
    "answer": "",
    "evaluate": "",
    "score": "",
    "check": ""
  }
]
```

---

## 11. Ví dụ Use Cases

### Use case 1: Dùng DeepSeek mặc định

```bash
python script.py --input document.md --num-questions 20
```

### Use case 2: Dùng OpenAI qua corporate proxy

```bash
python script.py \
  --input document.md \
  --provider openai \
  --api-key sk-xxx \
  --base-url https://corporate-proxy.company.com/openai/v1 \
  --num-questions 30
```

### Use case 3: Dùng Ollama local với model custom

```bash
python script.py \
  --input document.md \
  --provider ollama \
  --ollama-model "llama3.2:3b" \
  --num-questions 15
```

### Use case 4: Preview trước khi lưu

```bash
python script.py \
  --input document.md \
  --provider anthropic \
  --api-key sk-ant-xxx \
  --num-questions 25 \
  --preview
```

---

## 12. Ghi chú kỹ thuật

- Câu hỏi **chỉ được chấp nhận** nếu:
  - Có `answer_location`
  - Đoạn trả lời **match nguyên văn** trong tài liệu gốc
- Tool phù hợp để tạo **RAG evaluation dataset** (faithfulness / grounding)
- Hỗ trợ retry logic: Tối đa 3 lần thử nếu không đủ số câu hỏi

---

## 13. Hỗ trợ

Xem toàn bộ options:

```bash
python script.py --help
```

Ví dụ output:

```
usage: script.py [-h] --input INPUT [--output OUTPUT] [--num-questions NUM_QUESTIONS]
                 [--api-key API_KEY] [--provider {deepseek,openai,anthropic,ollama,api}]
                 [--ollama-model OLLAMA_MODEL] [--ollama-url OLLAMA_URL]
                 [--base-url BASE_URL] [--preview]

RAG Evaluation Generator - Vietnamese questions based on 4W1H
```
