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
python "script.py" --input "path\to\doc.txt"
```

- Provider mặc định: `deepseek`
- API key lấy từ biến môi trường `DEEPSEEK_API_KEY` nếu không truyền `--api-key`

---

## 3. Truyền API key bằng flag `--api-key`

### DeepSeek (default)

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --api-key YOUR_DEEPSEEK_KEY
```

### OpenAI

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider openai \
  --api-key YOUR_OPENAI_KEY
```

### Anthropic

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider anthropic \
  --api-key YOUR_ANTHROPIC_KEY
```

---

## 4. Dùng API key qua biến môi trường (.env hoặc system env)

### DeepSeek (default)

```bash
python "script.py" --input "path\to\doc.txt"
```

Yêu cầu biến môi trường:

```env
DEEPSEEK_API_KEY=sk-xxxx
```

---

### OpenAI

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider openai
```

```env
OPENAI_API_KEY=sk-xxxx
```

---

### Anthropic

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider anthropic
```

```env
ANTHROPIC_API_KEY=sk-ant-xxxx
```

---

## 5. Dùng Ollama (local LLM)

### Ollama – mặc định

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider ollama
```

Mặc định:
- URL: `http://localhost:11434`
- Model: lấy từ `OLLAMA_MODEL` hoặc config mặc định

---

### Chỉ định model và URL Ollama

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider ollama \
  --ollama-model "qwen2.5:7b" \
  --ollama-url "http://localhost:11434"
```

---

## 6. Alias provider `api`

Alias `api` sẽ map về **DeepSeek**:

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --provider api \
  --api-key YOUR_DEEPSEEK_KEY
```

---

## 7. Tuỳ chọn CLI thường dùng

```bash
--num-questions 30     # Số câu hỏi cần sinh
--output output.json   # File JSON đầu ra
--preview              # Xem trước 5 câu hỏi, hỏi có lưu hay không
```

Ví dụ:

```bash
python "script.py" \
  --input "path\to\doc.txt" \
  --num-questions 30 \
  --preview
```

---

## 8. Output

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

## 9. Ghi chú kỹ thuật

- Câu hỏi **chỉ được chấp nhận** nếu:
  - Có `answer_location`
  - Đoạn trả lời **match nguyên văn** trong tài liệu gốc
- Tool phù hợp để tạo **RAG evaluation dataset** (faithfulness / grounding)

---

## 10. Hỗ trợ

- Chạy `--help` để xem toàn bộ option:

```bash
python "script.py" --help
```

