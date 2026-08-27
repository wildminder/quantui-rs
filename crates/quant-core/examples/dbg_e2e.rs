use quant_core::manifest::QuantConfig;
use quant_core::stream::stream_quantize;
fn main() {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir = dir.join("tests/golden/linear_basic_bf16");
    let out = std::path::PathBuf::from("tmp_out2.safetensors");
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(out.with_extension("safetensors.quant-manifest.json"));
    let cfg = QuantConfig {
        exclude_layers: None,
        ..Default::default()
    };
    let r = stream_quantize(dir.join("input.safetensors"), &out, &cfg).unwrap();
    println!("done={} order={}", r.done.len(), r.order.len());
    let a = std::fs::read(&out).unwrap();
    let b = std::fs::read(dir.join("output.safetensors")).unwrap();
    if a == b {
        println!("BYTES EQUAL");
    } else {
        // locate first difference
        let n = a.len().min(b.len());
        for i in 0..n {
            if a[i] != b[i] {
                println!("first diff at byte {i} of {}", n);
                break;
            }
        }
        println!("len ours={} gold={}", a.len(), b.len());
    }
}
