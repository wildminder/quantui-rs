use quant_core::torch_rng::TorchRng;
use std::env;

fn main() {
    let n = 128usize;
    let seed = 233983427u64;
    let x = TorchRng::manual_seed(seed).randn_f32(3072 * n);
    let path = env::var("RNG_DUMP").unwrap_or_else(|_| "rust_x128.bin".to_string());
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).unwrap();
    println!("wrote {} floats ({}) to {}", x.len(), bytes.len(), path);
}
