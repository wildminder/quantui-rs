//! Debug: dump both headers to find the mismatch source.
use quant_core::st_io::reader::SafetensorsReader;
use std::path::PathBuf;

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden").join(name)
}

#[test]
fn dump_headers() {
    let dir = golden("linear_basic_bf16");
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.safetensors");
    quant_core::stream::stream_quantize(
        dir.join("input.safetensors"),
        &out,
        &quant_core::manifest::QuantConfig::default(),
    )
    .unwrap();

    let ours = SafetensorsReader::open(&out).unwrap();
    let theirs = SafetensorsReader::open(dir.join("output.safetensors")).unwrap();

    for (n, i) in theirs.header().iter() {
        let o = ours.header().get(n);
        println!(
            "{n}: GOLDEN dtype={} shape={:?} offs={:?} | OURS {}",
            i.dtype_raw,
            i.shape,
            i.data_offsets,
            o.map(|oi| format!(
                "dtype={} shape={:?} offs={:?}",
                oi.dtype_raw, oi.shape, oi.data_offsets
            ))
            .unwrap_or_else(|| "MISSING".into())
        );
    }
    println!(
        "slot ours={} theirs={}",
        u64::from_le_bytes(std::fs::read(&out).unwrap()[0..8].try_into().unwrap()),
        u64::from_le_bytes(
            std::fs::read(dir.join("output.safetensors")).unwrap()[0..8]
                .try_into()
                .unwrap()
        )
    );

    // Byte-diff first divergence
    let a = std::fs::read(&out).unwrap();
    let b = std::fs::read(dir.join("output.safetensors")).unwrap();
    println!("len ours={} golden={}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x != y {
            println!("first diff at byte {i}: ours={x:#04x} golden={y:#04x}");
            println!(
                "context ours: {:02x?}",
                &a[i.saturating_sub(16)..(i + 16).min(a.len())]
            );
            println!(
                "context gold: {:02x?}",
                &b[i.saturating_sub(16)..(i + 16).min(b.len())]
            );
            break;
        }
    }
}
