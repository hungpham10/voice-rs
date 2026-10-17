//! GĐ4 — huấn luyện **chỉ `Adapter`** trên một speaker đích (overfit 1 câu).
//!
//! `dec` (VITS HiFi-GAN) **đóng băng**; chỉ `Linear(768→192)` của `Adapter` học.
//! Loss = `L1` giữa log-mel của wav sinh ra và log-mel của wav đích, cả hai ở
//! 22.05 kHz (wav đích 16 kHz được resample lên ngoài mạng, §13.2.1).
//!
//! Đây là một **integration test**, không phải unit test — chạy thật, cần
//! `sources/vits_dec/` + wav + features:
//!
//! ```bash
//! python3 scripts/export_vits_dec.py
//! cargo run --release -p voice-hubert --example embed -- sources/422-122949-0031.wav sources/features/422-122949-0031.npy
//! cargo run --release -p voice-train  --example train_adapter
//! ```
//!
//! Biến môi trường (tuỳ chọn): `SEG` = số frame latent (mặc định cả câu),
//! `STEPS`, `LR`. Kỳ vọng: loss giảm rõ rệt, `test_output.wav` nghe giống giọng
//! đích.

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{Device, Tensor, TensorData};
use voice_lib::vc::{resample_linear, Adapter};
use voice_models::dec::Generator;
use voice_train::{data, mel};

const DEC_DIR: &str = "sources/vits_dec";
const WAV: &str = "sources/422-122949-0031.wav";
const FEATS: &str = "sources/features/422-122949-0031.npy";
const OUT_WAV: &str = "test_output.wav";
const OUT_ADAPTER: &str = "sources/features/adapter.bpk";

/// 6.84 s × 22050/256, làm tròn — §14.4.
const T_OUT: usize = 589;
const LR: f64 = 1e-3;
const STEPS: usize = 300;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dec_dir = PathBuf::from(DEC_DIR);
    if !dec_dir.join("config.json").exists() {
        return Err(format!("{DEC_DIR}/ chưa có — chạy scripts/export_vits_dec.py").into());
    }
    if !PathBuf::from(FEATS).exists() {
        return Err(format!("{FEATS} chưa có — chạy `embed <wav> <out.npy>` trước").into());
    }

    let device = Device::default().autodiff();

    // `dec` đóng băng: `freeze()` tắt require_grad toàn cây → không gradient nào
    // chảy vào 14.33 M tham số của nó (khớp GĐ2).
    let gen = Generator::load(&dec_dir, &device)?.freeze();
    // Chỉ adapter học. `unfreeze()` đặt require_grad = true cho `Linear`.
    let mut adapter = Adapter::new(&device).unfreeze();
    println!(
        "dec (đóng băng) = {:.2} M tham số · adapter (học) = {} tham số\n",
        gen.num_params() as f64 / 1e6,
        adapter.in_dim() * adapter.out_dim() + adapter.out_dim()
    );

    // ---- dữ liệu
    let cfg = voice_mel::MelConfig::ljs();
    let basis = mel::mel_basis(&cfg, &device);

    let feats_full = data::features_channel_major(&PathBuf::from(FEATS), &device)?; // [1,768,T_h]
    println!("features {:?}", feats_full.dims());

    // `SEG` = số frame latent để huấn luyện (mặc định cả câu). Đoạn ngắn để smoke
    // nhanh: `dec` forward+backward trên Flex CPU khá nặng ở 589 frame.
    let t_out: usize = std::env::var("SEG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(T_OUT);
    let t_h = ((t_out as f64 * 50.0 / 86.1328125).round() as usize).min(feats_full.dims()[2]);
    let feats = feats_full.narrow(2, 0, t_h); // [1,768,t_h]
    let n = t_out * 256; // wav sinh ra có đúng t_out·256 mẫu

    let (mut target_wav, sr) = data::read_wav_mono(&PathBuf::from(WAV))?;
    if sr != 16000 {
        eprintln!("⚠️  wav nguồn {sr} Hz, kỳ vọng 16000 Hz");
    }
    target_wav = resample_linear(&target_wav, sr, 22050);
    target_wav.truncate(n);
    let target = Tensor::<2>::from_data(TensorData::new(target_wav, [1, n]), &device);
    println!("huấn luyện {t_out} frame latent ({t_h} frame HuBERT) · đích {n} mẫu\n");

    // ---- huấn luyện
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(STEPS);
    let lr: f64 = std::env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(LR);

    let mut optim = burn::optim::AdamConfig::new().init();
    let mut first = f64::NAN;
    let mut last = f64::NAN;

    for step in 0..steps {
        let z = adapter.forward_to(feats.clone(), t_out); // [1,192,t_out]
        let wav = gen.forward(z).reshape([1, n]); // [1,n]
        let loss = mel::mel_l1(wav, target.clone(), &cfg, &basis); // Tensor<1>

        let lv: f64 = loss.clone().into_scalar::<f32>() as f64;
        if step == 0 {
            first = lv;
        }
        last = lv;

        let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &adapter);
        adapter = optim.step(lr, adapter, grads);

        if step % 25 == 0 || step == steps - 1 {
            println!("step {step:>3}  mel-L1 = {lv:.5}");
        }
    }

    let ratio = first / last;
    println!("\nmel-L1: {first:.5} → {last:.5}  (giảm {ratio:.2}×)");

    // ---- nghe thử: sinh wav cuối bằng adapter đã học
    let z = adapter.forward_to(feats, t_out);
    let wav = gen
        .forward(z)
        .reshape([1, n])
        .into_data()
        .as_slice::<f32>()?
        .to_vec();
    data::write_wav_mono(&PathBuf::from(OUT_WAV), &wav, 22050)?;
    adapter.save(&PathBuf::from(OUT_ADAPTER))?;
    println!("đã ghi {OUT_WAV} và {OUT_ADAPTER}");

    if !(ratio.is_finite() && ratio > 1.0) {
        eprintln!("\n❌ loss KHÔNG giảm — kiểm dữ liệu/lr");
        std::process::exit(1);
    }
    println!("\nPASS — loss giảm {ratio:.2}×");
    Ok(())
}
