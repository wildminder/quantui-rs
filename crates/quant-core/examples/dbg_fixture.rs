use quant_core::manifest::QuantConfig;
use quant_core::stream::stream_quantize;

fn main() {
    let name = std::env::args().nth(1).expect("fixture name");
    let exclude = std::env::args().nth(2);
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir = dir.join("tests/golden").join(&name);
    let out = std::path::PathBuf::from(format!("tmp_{}.safetensors", name));
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(out.with_extension("safetensors.quant-manifest.json"));
    let cfg = QuantConfig {
        exclude_layers: exclude.map(|s| s.to_string()),
        ..Default::default()
    };
    let _r = stream_quantize(dir.join("input.safetensors"), &out, &cfg).unwrap();
    let a = std::fs::read(&out).unwrap();
    let b = std::fs::read(dir.join("output.safetensors")).unwrap();
    if a == b {
        println!("BYTES EQUAL for {}", name);
        return;
    }
    let n = a.len().min(b.len());
    let mut first = n;
    for i in 0..n {
        if a[i] != b[i] {
            first = i;
            break;
        }
    }
    let hl_a = u64::from_le_bytes(a[0..8].try_into().unwrap());
    let hl_b = u64::from_le_bytes(b[0..8].try_into().unwrap());
    println!(
        "FIRST DIFF at byte {} (len ours={} gold={} header_ours={} header_gold={})",
        first,
        a.len(),
        b.len(),
        hl_a,
        hl_b
    );
}
