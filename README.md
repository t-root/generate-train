# Generate Train

Dự án gồm 2 luồng chính:
1) `generate`: sinh dữ liệu hội thoại, lọc trùng ngữ nghĩa, lưu ChromaDB  
2) `train`: fine-tune LoRA từ dữ liệu đã lọc, có dashboard theo dõi tiến trình

## Cấu trúc 

```text
generate-train/
├── .hf_cache/              # base model Hugging Face (tự tạo khi tải)
│   └── hub/models--.../
├── data/
│   ├── chroma_db/          # vector store + bản ghi QA
│   ├── finetuned_lora_*/   # LoRA sau train (tên tự thêm model name)
│   └── view_data.py        # xem/sửa dữ liệu (Flask)
├── generate/
│   ├── generate.py
│   ├── generate_config.json
│   ├── chroma_db.py
│   ├── checks.py
│   ├── prompt.txt
│   └── index.html
├── train/
│   ├── train.py
│   ├── train_config.json
│   └── index.html
├── test.py                 # chat thử model + LoRA
├── run.py
└── requirements.txt
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sửa `generate/generate_config.json` và `train/train_config.json` — thay `YOUR_API_URL`, `YOUR_API_KEY`, `YOUR_MODEL_NAME` bằng giá trị thật của bạn.

## Cách chạy nhanh

Chạy menu chọn chế độ:

```bash
python run.py
```

Menu:
- `1`: chạy `generate`
- `2`: chạy `train`
- `3`: xem/sửa dữ liệu (`data/view_data.py`)
- `4`: chat thử model đã train (`test.py`)

## Luồng generate

Chạy trực tiếp:

```bash
python -m generate.generate
```

Tùy chỉnh:

```bash
python -m generate.generate --target 200 --port 5000
```

Dashboard generate:
- `http://localhost:5000`
- API: `GET /api/stats`

## Luồng train

Chạy trực tiếp:

```bash
python train/train.py
```

Tùy chỉnh port dashboard:

```bash
python train/train.py --port 5001
```

Dashboard train:
- `http://localhost:5001`
- API: `GET /api/stats`

Lưu ý: `train/train.py` đọc mẫu `selected` từ ChromaDB (`data/chroma_db`) qua `data/view_data.py`.

## Cấu hình

### Cấu hình generate

File: `generate/generate_config.json`

Cấu trúc JSON:

```json
{
  "inference_mode": "api",        // "api" hoặc "local"
  "api": {
    "url": "YOUR_API_URL",
    "key": "YOUR_API_KEY",
    "model": "YOUR_MODEL_NAME",
    "timeout_seconds": 120,
    "reasoning_effort": "none",
    "reasoning_exclude": true
  },
  "local": {
    "model": "YOUR_MODEL_NAME"
  },
  "common": {
    "embed_model": "sentence-transformers/...",
    "target_count": 100,
    "samples_per_request": 1,
    "max_attempts": 5000,
    "max_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.95,
    "vector_sim_threshold": 0.999999,
    "chroma_collection": "qa_samples"
  }
}
```

**Lưu ý:** Tùy `inference_mode` là `"api"` hay `"local"`, chương trình sẽ dùng cấu hình tương ứng.

| Trường | Ý nghĩa |
|--------|---------|
| `target_count` | Tổng số mẫu `selected` cần tích lũy trong ChromaDB |
| `samples_per_request` | Số cặp QA mỗi lần gọi AI (mặc định `1`). Nếu > 1, model trả về mảng JSON; pipeline xử lý từng phần tử. Tăng `max_tokens` khi dùng giá trị lớn. |
| `max_attempts` | Số lần gọi AI tối đa (mỗi lần có thể trả về `samples_per_request` mẫu) |

### Cấu hình train

File: `train/train_config.json`

```json
{
  "model_name": "YOUR_MODEL_NAME",
  "require_cuda": false,
  "epochs": 3,
  "batch_size": 2,
  "cpu_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "learning_rate": 2e-4,
  "max_seq_length": 512,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "chroma_collection": "qa_samples"
}
```

**Output directory tự động thêm tên model:** nếu `model_name = "Qwen/Qwen2.5-1.5B-Instruct"` → output sẽ là `data/finetuned_lora_Qwen2.5-1.5B-Instruct`

## `checks.py` là chỗ custom logic

File: `generate/checks.py` là nơi để bạn tự định nghĩa rule kiểm tra bản ghi theo nhu cầu.

Hiện tại file này chỉ đang check rule cơ bản. Bạn có thể tùy ý mở rộng thêm, ví dụ:
- cấm từ khóa
- bắt buộc độ dài tối thiểu/tối đa
- chuẩn format riêng cho nghiệp vụ
- kiểm tra tone/đại từ xưng hô nâng cao

Nói ngắn gọn: muốn đổi luật lọc dữ liệu thì sửa `generate/checks.py` (giữ khớp từ khóa persona trong `generate/prompt.txt`).

## Cache model Hugging Face (base model tải về)

Cấu hình tự động trong `generate.py` và `train.py` (`HF_HOME` → `.hf_cache/`).

- Thư mục: `.hf_cache/hub/models--<tên-repo>/snapshots/<hash>/`
- Ví dụ Qwen train: `.hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/`

Model đã tải trước đó ở `%USERPROFILE%\.cache\huggingface\` **không** tự chuyển sang — copy thư mục `models--...` sang `.hf_cache/hub/` hoặc chạy lại để tải lại.

## File đầu ra quan trọng

- `data/chroma_db/`: dữ liệu QA + embedding (generate / train đọc từ đây)
- `data/finetuned_lora_<model_name>/`: LoRA adapter + tokenizer sau train (tên tự động dựa trên model)
  - Ví dụ: `data/finetuned_lora_Qwen2.5-1.5B-Instruct/`
- `.hf_cache/`: base model tải từ Hugging Face

## Lưu ý

- Lần chạy đầu có thể chậm do tải model vào `.hf_cache/`.
- Fine-tune 4-bit cần môi trường CUDA/bitsandbytes phù hợp; `require_cuda: false` cho phép train full precision trên CPU (chậm).
- Cấu hình hoàn toàn dựa trên JSON (`generate_config.json` và `train_config.json`) — chỉnh sửa cấu hình bằng sửa JSON thay vì code Python.
- Persona / ví dụ hội thoại: sửa trực tiếp `generate/prompt.txt`.
