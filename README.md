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

Cần Rust stable. Trọng số tải riêng (xem dưới).

```bash
cargo test -p voice-lib -p voice-mel -p voice-models -p voice-train   # CI chạy đúng bộ này

./scripts/fetch_assets.sh          # tải bundle model từ Release assets-v1 → sources/
./scripts/run_training.sh          # build → validation (verify_dec) → training adapter
```

## Model (không nằm trong git)

`sources/` bị `.gitignore`. Cách gọn nhất: `./scripts/fetch_assets.sh` tải bundle
**chỉ-model** từ Release [`assets-v1`](https://github.com/hungpham10/voice-rs/releases/tag/assets-v1):

| Bundle | Chứa |
|---|---|
| `voice-assets.tar.gz` | ContentVec ONNX + `vits_dec` (VITS generator đã export) |
| `f0-assets.tar.gz` (`--f0`) | `rmvpe.onnx` (cho `voice-f0`) |

**Không cần PyTorch** trên máy đích (export đã làm trong CI). Bundle kèm `SHA256SUMS`
để `fetch_assets.sh` tự verify. Dữ liệu đích (dataset) tự tải khi cần.

Nguồn pull gốc cho workflow nằm ở [`scripts/asset_checksums.txt`](scripts/asset_checksums.txt):
ContentVec từ HF riêng `hung0913208/voice-rs-assets`, RMVPE từ `TigreGotico/voiceclonnx-rvc`,
checkpoint VITS từ Google Drive — mỗi file kèm sha256 để verify.

Hoặc dùng thẳng notebook `notebooks/colab_train.ipynb` — nó gọi 2 script trên.

## Giấy phép

MIT — xem [`LICENSE`](LICENSE).

Repo port kiến trúc VITS HiFi-GAN generator sang Rust/Burn.
`samples/422-122949-0031.wav` trích từ **LibriSpeech** (`test-clean`, reader 422),
giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
