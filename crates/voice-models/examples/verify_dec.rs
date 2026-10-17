//! Verify generator VITS (`dec`) viết tay bằng Burn với PyTorch.
//!
//! Đối chiếu `sources/vits_dec/ref_audio.npz` (sinh bởi
//! `scripts/export_vits_dec.py`) trên 7 giá trị `T`:
//!   - biên dưới : T = 1, 2      → bắt lỗi ở kích thước tối thiểu
//!   - lẻ        : T = 37, 137   → lỗi off-by-one của ConvTranspose chỉ lộ ở đây
//!   - tròn      : T = 10, 100, 250
//!
//! Run: cargo run --release -p voice-models --example verify_dec

use std::path::PathBuf;

use burn::tensor::{Device, Tensor, TensorData};
use voice_models::dec::Generator;

mod common;
use common::read_npz;

const DIR: &str = "sources/vits_dec";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(DIR);
    let audio_npz = dir.join("ref_audio.npz");
    let input_npz = dir.join("ref_input.npz");
    if !audio_npz.exists() {
        return Err(format!("{audio_npz:?} chưa có — chạy scripts/export_vits_dec.py").into());
    }

    let device = Device::default();
    println!("Nạp generator từ {DIR}/dec/ ...");
    let gen = Generator::load(&dir, &device)?;
    println!("  OK\n");

    let ref_a = read_npz(&audio_npz)?;
    let ref_z = read_npz(&input_npz)?;

    let mut keys: Vec<&String> = ref_a.keys().collect();
    keys.sort_by_key(|k| k.trim_start_matches('T').parse::<usize>().unwrap_or(0));

    let mut all_pass = true;
    let mut worst_overall = 0.0f64;

    println!(
        "{:>5}  {:>10}  {:>10}  {:>12}  {:>11}  {}",
        "T", "samples", "mong doi", "max abs diff", "rms", "ket qua"
    );

    for k in &keys {
        let t: usize = k.trim_start_matches('T').parse()?;
        let expect_n = t * 256;
        let (z_flat, _z_dims) = ref_z.get(*k).expect("thieu z");
        let (ref_flat, _) = ref_a.get(*k).expect("thieu audio");

        let z = Tensor::from_data(
            TensorData::new(z_flat.clone(), [1usize, 192, t]),
            &device,
        );
        let out = gen.forward(z);
        let [b, c, n] = out.dims();

        if b != 1 || c != 1 {
            println!("T={t:<5} shape {b}x{c}x{n} — SAI shape, bo qua");
            all_pass = false;
            continue;
        }
        if n != expect_n {
            println!("T={t:<5} {n:>10}  {expect_n:>10}  {:>12}  {:>11}  FAIL chieu dai",
                "-", "-");
            all_pass = false;
            continue;
        }

        let got = out.into_data().as_slice::<f32>()?.to_vec();

        let mut max_abs = 0.0f64;
        let mut sq = 0.0f64;
        let mut sq_ref = 0.0f64;
        for (a, b) in got.iter().zip(ref_flat.iter()) {
            let d = (*a as f64 - *b as f64).abs();
            if d > max_abs {
                max_abs = d;
            }
            sq += (*a as f64).powi(2);
            sq_ref += (*b as f64).powi(2);
        }
        let rms_got = (sq / got.len() as f64).sqrt();
        let rms_ref = (sq_ref / ref_flat.len() as f64).sqrt();
        worst_overall = worst_overall.max(max_abs);

        let ok = max_abs < 1e-4 && (rms_got - rms_ref).abs() < 1e-5;
        all_pass &= ok;
        println!(
            "{t:>5}  {n:>10}  {expect_n:>10}  {max_abs:>12.3e}  {rms_got:>11.6}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!("\nmax abs diff lớn nhất: {worst_overall:.3e}");
    {
        let (data, _) = ref_a.values().next().expect("khong co mau nao");
        let sq: f64 = data.iter().map(|v| (*v as f64).powi(2)).sum();
        println!(
            "rms chuỗi đầu (PyTorch): {:.6}   n = {}",
            (sq / data.len() as f64).sqrt(),
            data.len()
        );
    }
    println!("\n{}", if all_pass { "PASS tat ca" } else { "CO FAIL" });
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}
