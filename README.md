# voice-rs

Hệ **voice conversion 1→1** (một giọng nguồn → một giọng đích) viết bằng **Rust /
[Burn](https://github.com/tracel-ai/burn)**, chạy suy luận **CPU-first**, kèm phụ đề
(closed caption) neo theo mốc thời gian của video nguồn.

> Inference hoàn toàn bằng Rust. Python/PyTorch chỉ dùng **offline** để export trọng
> số và tạo ground truth — không nằm trong runtime.

Tài liệu kiến trúc đầy đủ (nguồn sự thật duy nhất): **[`docs/architecture.md`](docs/architecture.md)**.

## Trạng thái

| Crate | Trách nhiệm | Trạng thái |
|---|---|---|
| `voice-hubert` | ContentVec 768 @ 50 Hz | ✅ max abs diff `3e-4` |
| `voice-mel` | log-mel (nhiều variant) | ✅ `4.77e-07` |
| `voice-f0` | RMVPE, f0 @ 100 Hz | ✅ mean diff `1.78 Hz` |
| `voice-models` | VITS HiFi-GAN generator (`dec`), viết tay | ✅ `1.26e-08` vs PyTorch |
| `voice-lib` | Thư viện dùng chung: `cc` (dấu câu, frame, `.srt`) + `vc` (`Adapter`, resample) | 🟡 43 unit test |
| `voice-train` | Huấn luyện/validate adapter (log-mel khả vi bằng Burn) | 🟡 loss giảm `2.22×`/40 bước |

## Cấu trúc

```
crates/
├── voice-hubert/   ContentVec embedder (ONNX) — 768 @ 50 Hz
├── voice-mel/      log-mel variants (rosa-based)
├── voice-f0/       RMVPE f0 @ 100 Hz (ONNX)
├── voice-models/   VITS HiFi-GAN `dec` viết tay bằng Burn
├── voice-lib/      thư viện dùng chung (cho bên ngoài)
└── voice-train/    huấn luyện + kiểm định (không kéo vào voice-lib)
docs/architecture.md   nguồn sự thật
notebooks/colab_train.ipynb   huấn luyện adapter trên Colab (CPU)
samples/               audio test nhỏ, commit được
scripts/               export trọng số / ground truth (Python, offline)
```

## Chạy nhanh (local)

Cần Rust stable. Trọng số + ground truth tải riêng (xem dưới).

```bash
cargo build --workspace
cargo test -p voice-lib -p voice-mel -p voice-train

# kiểm tra generator so với PyTorch
cargo run --release -p voice-models --example verify_dec
```

## Asset nặng (không nằm trong git)

`sources/` bị `.gitignore`. Cần các file sau để chạy đầy đủ:

| File | Lấy từ đâu |
|---|---|
| `sources/pretrained_ljs.pth` (139 MB) | [Google Drive của dự án VITS gốc](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2) hoặc Release `assets-v1` của repo này |
| `sources/onnx_hubert_new/model.onnx` (360 MB) | Release `assets-v1` (ContentVec-768, `HubertModelWithFinalProj`) |

Hoặc dùng thẳng notebook `notebooks/colab_train.ipynb` — nó tải asset, export, sinh
features và huấn luyện.

## Giấy phép

MIT — xem [`LICENSE`](LICENSE).

Repo port kiến trúc VITS HiFi-GAN generator sang Rust/Burn.
`samples/422-122949-0031.wav` trích từ **LibriSpeech** (`test-clean`, reader 422),
giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
