// build.rs - Generate Rust code from ONNX model using burn-onnx

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun if the embedding flag changes
    println!("cargo:rerun-if-env-changed=VOICE_EMBED_WEIGHTS");
    // Register the custom cfg (Rust 1.80+ check-cfg lint).
    println!("cargo:rustc-check-cfg=cfg(embed_weights)");

    // Tell cargo to rerun if the ONNX model changes
    let onnx_path = PathBuf::from("../../sources/contentvec/contentvec_768l12.onnx");
    println!("cargo:rerun-if-changed={}", onnx_path.display());

    if env::var("VOICE_EMBED_WEIGHTS").as_deref() == Ok("1") {
        // Bake the weights into the binary so it is self-contained (no `.bpk`
        // file needed at runtime). Used when packaging distributable binaries.
        println!("cargo:rustc-cfg=embed_weights");
    }

    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if !onnx_path.exists() {
        eprintln!(
            "ERROR: ONNX model not found at {}",
            onnx_path.display()
        );
        eprintln!("Run ./scripts/fetch_assets.sh to download it from HuggingFace.");
        std::process::exit(1);
    }

    // Generate Rust code from ONNX model.
    // LoadStrategy::File keeps the weights in a `.bpk` file; when
    // `VOICE_EMBED_WEIGHTS=1` the code below `include_bytes!`s that file instead.
    burn_onnx::ModelGen::new()
        .input(onnx_path.to_str().expect("ONNX path not valid UTF-8"))
        .out_dir("model")
        .load_strategy(burn_onnx::LoadStrategy::File)
        .run_from_script();
}
