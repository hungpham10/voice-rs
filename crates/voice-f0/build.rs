// build.rs - Generate Rust code from ONNX model using burn-onnx

use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun if the ONNX model changes
    let onnx_path = PathBuf::from("../../sources/rmvpe.onnx");
    println!("cargo:rerun-if-changed={}", onnx_path.display());

    // Check if model exists
    if !onnx_path.exists() {
        eprintln!(
            "ERROR: ONNX model not found at {}",
            onnx_path.display()
        );
        std::process::exit(1);
    }

    // Generate Rust code from ONNX model
    // Use LoadStrategy::File to keep weights in separate .bpk file
    burn_onnx::ModelGen::new()
        .input(onnx_path.to_str().expect("ONNX path not valid UTF-8"))
        .out_dir("model")
        .load_strategy(burn_onnx::LoadStrategy::File)
        .run_from_script();
}