//! Phase 1 tests: header round-trip, golden parity, kill-simulation resume,
//! UINT16 tolerance, malformed-input handling, official-crate cross-check.

use std::path::PathBuf;

use quant_core::dtype::DType;
use quant_core::st_io::{Header, IncrementalWriter, SafetensorsReader};

fn golden(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.join("tests/golden").join(name)
}

fn all_goldens() -> Vec<(&'static str, bool)> {
    vec![
        ("linear_basic_bf16/input.safetensors", false),
        ("linear_basic_bf16/output.safetensors", true),
        ("odd_shapes/input.safetensors", false),
        ("odd_shapes/output.safetensors", true),
        ("conv_net/input.safetensors", false),
        ("conv_net/output.safetensors", true),
    ]
}

// ---------------------------------------------------------------- 1.1 parse

#[test]
fn parse_all_golden_files() {
    for (rel, _is_output) in all_goldens() {
        let path = golden(rel);
        let r = SafetensorsReader::open(&path).unwrap_or_else(|e| panic!("open {rel}: {e}"));
        assert!(r.header().len() > 0, "{rel}: no tensors");
        // Every tensor's offsets must be in-range (validated at open).
        assert!(!r.path().as_os_str().is_empty());
    }
}

#[test]
fn linear_basic_bf16_expected_content() {
    let r = SafetensorsReader::open(golden("linear_basic_bf16/output.safetensors")).unwrap();
    let h = r.header();
    // Quantized weight + scale + comfy_quant + input_scale + corrected bias.
    let w = h.get("blocks.0.weight").expect("blocks.0.weight");
    assert_eq!(w.dtype_raw, "I8");
    assert_eq!(w.shape, vec![256, 128]);
    let s = h.get("blocks.0.weight_scale").expect("weight_scale");
    assert_eq!(s.dtype_raw, "F32");
    // Scale-squeeze quirk: [2,1] block grid... actually [256/128,128/128]=[2,1].
    assert_eq!(s.shape, vec![2, 1]);
    let cq = h.get("blocks.0.comfy_quant").expect("comfy_quant");
    assert_eq!(cq.dtype_raw, "U8");
    assert_eq!(cq.shape.len(), 1);
    let is = h.get("blocks.0.input_scale").expect("input_scale");
    assert_eq!(is.dtype_raw, "F32");
    assert!(is.shape.is_empty(), "input_scale must be scalar shape []");
    assert!(h.contains_key("blocks.0.bias"), "corrected bias present");
}

#[test]
fn odd_shapes_skip_and_exclude_quirks() {
    let inp = SafetensorsReader::open(golden("odd_shapes/input.safetensors")).unwrap();
    let out = SafetensorsReader::open(golden("odd_shapes/output.safetensors")).unwrap();
    // [130,130] tensor copied unchanged → raw bytes identical.
    let in_b = inp.tensor_bytes("odd.weight").unwrap();
    let out_b = out.tensor_bytes("odd.weight").unwrap();
    assert_eq!(
        in_b, out_b,
        "[130,130] skip-heur copy must be byte-identical"
    );
    // Excluded attn_norm keeps ORIGINAL dtype (F32), not cast to bf16.
    let norm_in = inp.header().get("attn_norm.weight").unwrap();
    let norm_out = out.header().get("attn_norm.weight").unwrap();
    assert_eq!(norm_in.dtype_raw, norm_out.dtype_raw);
}

// ------------------------------------------------------- 1.1 header round-trip

#[test]
fn header_roundtrip_exact_bytes_for_all_goldens() {
    for (rel, _) in all_goldens() {
        let path = golden(rel);
        let bytes = std::fs::read(&path).unwrap();
        let slot = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr_bytes = &bytes[8..8 + slot];
        let h = Header::parse_json_bytes(trim_sp(hdr_bytes), &path).expect("parse {rel}");
        let reser = h.serialize_json();
        assert_eq!(
            trim_sp(hdr_bytes),
            trim_sp(&reser),
            "{rel}: re-serialized JSON must equal original (modulo padding)"
        );
    }
}

fn trim_sp(b: &[u8]) -> &[u8] {
    let mut e = b.len();
    while e > 0 && b[e - 1] == b' ' {
        e -= 1;
    }
    &b[..e]
}

// ------------------------------------------------------------ 1.3 writer parity

#[test]
fn writer_rebuilds_golden_outputs_byte_identically() {
    for (rel, is_output) in all_goldens() {
        if !is_output {
            continue;
        }
        let path = golden(rel);
        let src = SafetensorsReader::open(&path).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let out_path = tmp.path().join("rebuild.safetensors");

        let mut w = IncrementalWriter::open_new(&out_path).unwrap();
        let mut order: Vec<(&String, &quant_core::st_io::header::TensorInfo)> =
            src.header().iter().collect();
        // Re-append in file order (insertion order preserved by Header).
        order.sort_by_key(|(_, info)| info.data_offsets.0);
        for (name, info) in &order {
            let data = src.tensor_bytes(name).unwrap();
            w.add_tensor(
                name,
                info.dtype.clone(),
                Some(&info.dtype_raw),
                &info.shape,
                data,
            )
            .unwrap();
        }
        w.finalize().unwrap();

        let built = std::fs::read(&out_path).unwrap();
        let expected = std::fs::read(&path).unwrap();
        assert_eq!(built.len(), expected.len(), "{rel}: rebuilt size differs");
        assert_eq!(&built[..], &expected[..], "{rel}: rebuilt bytes differ");
    }
}

// ------------------------------------------------------------ 1.4 kill-sim

#[test]
fn kill_sim_resume_from_own_half_written_file() {
    let src = SafetensorsReader::open(golden("linear_basic_bf16/output.safetensors")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("partial.safetensors");

    let mut order: Vec<(&String, &quant_core::st_io::header::TensorInfo)> =
        src.header().iter().collect();
    order.sort_by_key(|(_, i)| i.data_offsets.0);

    // "Crash" after the first tensor.
    {
        let mut w = IncrementalWriter::open_new(&out_path).unwrap();
        let (n0, i0) = order[0];
        w.add_tensor(
            n0,
            i0.dtype.clone(),
            Some(&i0.dtype_raw),
            &i0.shape,
            src.tensor_bytes(n0).unwrap(),
        )
        .unwrap();
        drop(w); // no finalize — simulate hard kill
    }

    // Resume: skipped names are no-ops; remaining tensors appended.
    let mut w = IncrementalWriter::open_resume(&out_path).unwrap();
    assert_eq!(w.done(), &["blocks.0.bias"]);
    for (name, info) in &order {
        w.add_tensor(
            name,
            info.dtype.clone(),
            Some(&info.dtype_raw),
            &info.shape,
            src.tensor_bytes(name).unwrap(),
        )
        .unwrap();
    }
    w.finalize().unwrap();

    let built = std::fs::read(&out_path).unwrap();
    let expected = std::fs::read(golden("linear_basic_bf16/output.safetensors")).unwrap();
    assert_eq!(
        built, expected,
        "resumed file must equal golden byte-for-byte"
    );
}

#[test]
fn kill_sim_resume_from_python_truncated_partial() {
    // Take a golden output, truncate it mid-data-section (after slot + first
    // tensor payload) WITHOUT a valid final header — this mirrors what a
    // Python-process kill leaves on disk: partial data, last flushed header
    // listing only completed tensors.
    let golden_path = golden("conv_net/output.safetensors");
    let full = std::fs::read(&golden_path).unwrap();
    let slot = u64::from_le_bytes(full[0..8].try_into().unwrap()) as usize;

    // Find the end of the FIRST tensor payload from the parsed header.
    let reader_probe = SafetensorsReader::open(&golden_path).unwrap();
    let first_end = reader_probe
        .header()
        .iter()
        .map(|(_, i)| i.data_offsets.1)
        .min()
        .unwrap();

    // Truncated file: prefix up to just past the first tensor's payload.
    // The flushed header still lists ALL tensors, but data region is short.
    // Python's resume tolerates this by clamping negative data_len to 0; our
    // open_resume computes data_len = size - (8+slot) which equals
    // first_end here — meaning only the first tensor's bytes exist. To make
    // resume well-defined we emulate the real crash contract: rebuild the
    // header to list ONLY the first tensor (what the reference flush loop
    // would have last written before the kill).
    let truncated_header = build_partial_header(&reader_probe, first_end);

    let mut partial = Vec::new();
    partial.extend_from_slice(&(slot as u64).to_le_bytes());
    partial.extend_from_slice(&truncated_header);
    partial.extend_from_slice(&full[8 + slot..8 + slot + first_end as usize]);

    let tmp = tempfile::tempdir().unwrap();
    let part_path = tmp.path().join("python_crash.safetensors");
    std::fs::write(&part_path, &partial).unwrap();

    // Resume and complete.
    let mut w = IncrementalWriter::open_resume(&part_path).unwrap();
    let src = SafetensorsReader::open(&golden_path).unwrap();
    let mut order: Vec<(&String, &quant_core::st_io::header::TensorInfo)> =
        src.header().iter().collect();
    order.sort_by_key(|(_, i)| i.data_offsets.0);
    for (name, info) in &order {
        w.add_tensor(
            name,
            info.dtype.clone(),
            Some(&info.dtype_raw),
            &info.shape,
            src.tensor_bytes(name).unwrap(),
        )
        .unwrap();
    }
    w.finalize().unwrap();

    let built = std::fs::read(&part_path).unwrap();
    // Byte-parity requires identical slot layout; our writer reuses the slot
    // from the resumed file (65536) so output matches the golden exactly.
    assert_eq!(built, full, "resume-from-python-crash must equal golden");
}

/// Build a header JSON body listing only tensors whose payload fully fits in
/// `data_len` bytes, with offsets rewritten compactly (no gaps).
fn build_partial_header(reader: &SafetensorsReader, data_len: u64) -> Vec<u8> {
    let mut kept: Vec<(String, quant_core::st_io::header::TensorInfo)> = Vec::new();
    let mut cursor = 0u64;
    let mut order: Vec<(&String, &quant_core::st_io::header::TensorInfo)> =
        reader.header().iter().collect();
    order.sort_by_key(|(_, i)| i.data_offsets.0);
    for (name, info) in order {
        let len = info.data_offsets.1 - info.data_offsets.0;
        if cursor + len <= data_len {
            kept.push((
                name.clone(),
                quant_core::st_io::header::TensorInfo {
                    dtype: info.dtype.clone(),
                    dtype_raw: info.dtype_raw.clone(),
                    shape: info.shape.clone(),
                    data_offsets: (cursor, cursor + len),
                },
            ));
            cursor += len;
        }
    }
    let mut h = Header::default();
    for (n, i) in kept {
        h.insert(n, i);
    }
    let mut bytes = h.serialize_json();
    while bytes.len() % 8 != 0 {
        bytes.push(b' ');
    }
    // Keep the SAME total slot size as the original so resume reproduces the
    // golden layout byte-for-byte.
    let orig_slot = {
        let f = std::fs::File::open(reader.path()).unwrap();
        use std::io::Read;
        let mut lb = [0u8; 8];
        let mut f = f;
        f.read_exact(&mut lb).unwrap();
        u64::from_le_bytes(lb) as usize
    };
    bytes.resize(orig_slot, b' ');
    bytes
}

// -------------------------------------------------------- UINT16 tolerance

#[test]
fn uint16_header_tolerated() {
    let json = br#"{"w":{"dtype":"U16","shape":[2],"data_offsets":[0,4]}}"#;
    let mut padded = json.to_vec();
    while padded.len() % 8 != 0 {
        padded.push(b' ');
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(padded.len() as u64).to_le_bytes());
    file.extend_from_slice(&padded);
    file.extend_from_slice(&[1, 0, 0, 0]); // 4 payload bytes

    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("u16.safetensors");
    std::fs::write(&p, &file).unwrap();

    let r = SafetensorsReader::open(&p).unwrap();
    let info = r.header().get("w").unwrap();
    assert_eq!(info.dtype, DType::U16);
    assert_eq!(r.tensor_bytes("w").unwrap(), &[1, 0, 0, 0]);
}

// ------------------------------------------------------ malformed inputs

#[test]
fn malformed_inputs_yield_errors_not_panics() {
    let tmp = tempfile::tempdir().unwrap();

    // Empty file.
    let p_empty = tmp.path().join("empty.st");
    std::fs::write(&p_empty, b"").unwrap();
    assert!(SafetensorsReader::open(&p_empty).is_err());

    // Truncated length prefix.
    let p_short = tmp.path().join("short.st");
    std::fs::write(&p_short, [1, 2, 3]).unwrap();
    assert!(SafetensorsReader::open(&p_short).is_err());

    // Slot value misaligned / insane.
    let p_bad = tmp.path().join("badslot.st");
    let mut bad = vec![];
    bad.extend_from_slice(&7u64.to_le_bytes()); // not 8-aligned
    bad.extend_from_slice(&vec![b' '; 16]);
    std::fs::write(&p_bad, &bad).unwrap();
    match SafetensorsReader::open(&p_bad) {
        Err(quant_core::st_io::Error::InvalidSlot { .. }) => {}
        other => panic!("expected InvalidSlot, got {:?}", other.map(|_| ())),
    }

    // Header that is not JSON.
    let p_nj = tmp.path().join("notjson.st");
    let mut nj = vec![];
    nj.extend_from_slice(&16u64.to_le_bytes());
    nj.extend_from_slice(b"not json at all!");
    std::fs::write(&p_nj, &nj).unwrap();
    assert!(matches!(
        SafetensorsReader::open(&p_nj),
        Err(quant_core::st_io::Error::HeaderJson { .. })
    ));

    // Resume of too-small file.
    let p_rs = tmp.path().join("small_resume.st");
    std::fs::write(&p_rs, [0u8; 4]).unwrap();
    assert!(IncrementalWriter::open_resume(&p_rs).is_err());

    // Unknown dtype string. Header must be padded to >= 8 and slot valid;
    // JSON parses but dtype "Q9" is unknown → UnknownDtype.
    let p_ud = tmp.path().join("unkdt.st");
    let json = br#"{"w":{"dtype":"Q9","shape":[1],"data_offsets":[0,1]}}"#;
    let mut hdr = json.to_vec();
    while hdr.len() % 8 != 0 {
        hdr.push(b' ');
    }
    let mut ud = vec![];
    ud.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    ud.extend_from_slice(&hdr);
    ud.extend_from_slice(&[0u8]);
    std::fs::write(&p_ud, &ud).unwrap();
    assert!(matches!(
        SafetensorsReader::open(&p_ud),
        Err(quant_core::st_io::Error::UnknownDtype { .. })
    ));
}

// ------------------------------------------------ official-crate cross-check

#[test]
fn safetensors_crate_reads_our_writer_output() {
    let src = SafetensorsReader::open(golden("linear_basic_bf16/output.safetensors")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("xcheck.safetensors");

    let mut w = IncrementalWriter::open_new(&out_path).unwrap();
    let mut order: Vec<(&String, &quant_core::st_io::header::TensorInfo)> =
        src.header().iter().collect();
    order.sort_by_key(|(_, i)| i.data_offsets.0);
    for (name, info) in &order {
        w.add_tensor(
            name,
            info.dtype.clone(),
            Some(&info.dtype_raw),
            &info.shape,
            src.tensor_bytes(name).unwrap(),
        )
        .unwrap();
    }
    w.finalize().unwrap();

    let bytes = std::fs::read(&out_path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .expect("official crate must deserialize incremental-writer output");
    assert_eq!(st.len(), src.header().len());
}
