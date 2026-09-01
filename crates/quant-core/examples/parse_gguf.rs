//! Debug example: parse a GGUF file with rlx-gguf (independent parser)
//! and dump metadata + tensor names, to cross-check fixture files.
//! Usage: cargo run -p quant-core --example parse_gguf -- <file.gguf>

use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: parse_gguf <file>");
    match rlx_gguf::GgufFile::from_path(Path::new(&path)) {
        Ok(f) => {
            println!("version: {}, alignment: {}", f.version, f.alignment);
            for (k, v) in &f.metadata {
                println!("meta: {k} = {:?}", shorten(v));
            }
            for (name, t) in &f.tensors {
                println!(
                    "tensor: {name} dtype={:?} shape={:?} offset={}",
                    t.dtype, t.shape, t.offset
                );
            }
        }
        Err(e) => {
            eprintln!("PARSE FAILED: {e:#}");
            std::process::exit(1);
        }
    }
}

fn shorten(v: &rlx_gguf::MetaValue) -> String {
    match v {
        rlx_gguf::MetaValue::String(s) => s.clone(),
        rlx_gguf::MetaValue::Array(items) => format!("[{} items]", items.len()),
        other => format!("{other:?}"),
    }
}
