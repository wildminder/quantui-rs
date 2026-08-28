//! Header parse/serialize with insertion-order preservation and space padding.

use serde_json::{Map, Value};

use super::error::{Error, Result};
use super::HEADER_ALIGN;
use crate::dtype::DType;
use std::path::Path;

/// One tensor descriptor from a safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: DType,
    pub dtype_raw: String,
    pub shape: Vec<u64>,
    /// [start, end] byte offsets relative to the start of the data region.
    pub data_offsets: (u64, u64),
}

/// Parsed safetensors header. Preserves insertion order of tensor names
/// (`serde_json` `preserve_order` feature) — the reference writer never sorts.
#[derive(Debug, Clone, Default)]
pub struct Header {
    tensors: Vec<(String, TensorInfo)>,
    metadata: Option<Map<String, Value>>,
}

impl Header {
    /// Parse the JSON body of a header (already stripped of length prefix and
    /// space padding).
    pub fn parse_json_bytes(raw: &[u8], path: &Path) -> Result<Header> {
        let text = std::str::from_utf8(raw).map_err(|e| Error::HeaderJson {
            path: path.to_path_buf(),
            message: format!("not valid UTF-8: {e}"),
        })?;
        let value: Value = serde_json::from_str(text).map_err(|e| Error::HeaderJson {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let obj = match value {
            Value::Object(m) => m,
            _ => {
                return Err(Error::HeaderJson {
                    path: path.to_path_buf(),
                    message: "header is not a JSON object".into(),
                })
            }
        };

        // __metadata__ is optional; remaining keys are tensors in order.
        // NOTE: do NOT use `obj.remove("__metadata__")` — with the
        // `preserve_order` feature the Map is IndexMap-backed and `remove`
        // swap-removes, which scrambles tensor order when `__metadata__` is
        // the FIRST key (as in the MXFP8/NVFP4 goldens). Iterate and skip
        // instead, so tensor order is preserved regardless of where
        // `__metadata__` sits.
        let mut metadata: Option<Map<String, Value>> = None;
        let mut tensors = Vec::with_capacity(obj.len());
        for (name, entry) in obj {
            if name == "__metadata__" {
                metadata = Some(match entry {
                    Value::Object(m) => m,
                    other => Map::from_iter([("__invalid__".to_string(), other)]),
                });
                continue;
            }
            let info = Self::parse_entry(&name, &entry, path)?;
            tensors.push((name, info));
        }
        Ok(Header { tensors, metadata })
    }

    fn parse_entry(name: &str, entry: &Value, path: &Path) -> Result<TensorInfo> {
        let obj = entry.as_object().ok_or_else(|| Error::BadTensorEntry {
            path: path.to_path_buf(),
            name: name.to_string(),
            message: "entry is not an object".into(),
        })?;
        let dtype_str = obj
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::BadTensorEntry {
                path: path.to_path_buf(),
                name: name.to_string(),
                message: "missing or non-string 'dtype'".into(),
            })?
            .to_string();
        let shape = obj
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::BadTensorEntry {
                path: path.to_path_buf(),
                name: name.to_string(),
                message: "missing or non-array 'shape'".into(),
            })?
            .iter()
            .map(|v| {
                v.as_u64().ok_or_else(|| Error::BadTensorEntry {
                    path: path.to_path_buf(),
                    name: name.to_string(),
                    message: "'shape' contains a non-integer".into(),
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let offs = obj
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::BadTensorEntry {
                path: path.to_path_buf(),
                name: name.to_string(),
                message: "missing or non-array 'data_offsets'".into(),
            })?;
        if offs.len() != 2 {
            return Err(Error::BadTensorEntry {
                path: path.to_path_buf(),
                name: name.to_string(),
                message: "'data_offsets' must have exactly 2 elements".into(),
            });
        }
        let start = offs[0].as_u64().ok_or_else(|| Error::BadTensorEntry {
            path: path.to_path_buf(),
            name: name.to_string(),
            message: "'data_offsets[0]' is not an integer".into(),
        })?;
        let end = offs[1].as_u64().ok_or_else(|| Error::BadTensorEntry {
            path: path.to_path_buf(),
            name: name.to_string(),
            message: "'data_offsets[1]' is not an integer".into(),
        })?;

        let dtype = DType::from_header_str(&dtype_str).ok_or_else(|| Error::UnknownDtype {
            path: path.to_path_buf(),
            dtype: dtype_str.clone(),
        })?;

        Ok(TensorInfo {
            dtype,
            dtype_raw: dtype_str,
            shape,
            data_offsets: (start, end),
        })
    }

    /// Byte size implied by shape × dtype elem size.
    pub fn expected_size(info: &TensorInfo) -> Option<u64> {
        info.dtype
            .elem_size()
            .map(|esz| info.shape.iter().product::<u64>().saturating_mul(esz))
    }

    /// Serialize to compact JSON bytes (no sort_keys — preserves insertion
    /// order), matching Python `json.dumps(header, separators=(",", ":"))`.
    ///
    /// `__metadata__` is emitted FIRST when present, matching the reference
    /// `save_file` output (golden-verified for the MXFP8/NVFP4 goldens). INT8
    /// streaming never carries metadata, so this ordering is a no-op for the
    /// INT8 whole-file parity contract.
    ///
    /// NOTE on float formatting: headers produced by the reference pipeline
    /// contain no floats (dtypes are strings, shapes/offsets integers,
    /// metadata values are strings), so serde_json's integer formatting
    /// matches Python's for our golden corpus.
    pub fn serialize_json(&self) -> Vec<u8> {
        let mut obj = Map::new();
        if let Some(meta) = &self.metadata {
            obj.insert("__metadata__".into(), Value::Object(meta.clone()));
        }
        for (name, info) in &self.tensors {
            let mut entry = Map::new();
            entry.insert("dtype".into(), Value::String(info.dtype_raw.clone()));
            entry.insert(
                "shape".into(),
                Value::Array(info.shape.iter().map(|&d| Value::from(d)).collect()),
            );
            entry.insert(
                "data_offsets".into(),
                Value::Array(vec![
                    Value::from(info.data_offsets.0),
                    Value::from(info.data_offsets.1),
                ]),
            );
            obj.insert(name.clone(), Value::Object(entry));
        }
        // Compact separators: serde_json::to_vec uses no extra whitespace by default.
        serde_json::to_vec(&Value::Object(obj)).expect("header serialization cannot fail")
    }

    /// Pad serialized header bytes with spaces so total length % 8 == 0.
    /// Port of `_align_header_to_8`.
    pub fn align_to_8(mut bytes: Vec<u8>) -> Vec<u8> {
        let pad = (HEADER_ALIGN - (bytes.len() % HEADER_ALIGN)) % HEADER_ALIGN;
        bytes.extend(std::iter::repeat_n(b' ', pad));
        bytes
    }

    // -- accessors -------------------------------------------------------- //

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorInfo)> {
        self.tensors.iter().map(|(n, i)| (n, i))
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, i)| i)
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.iter().map(|(n, _)| n)
    }

    pub fn metadata(&self) -> Option<&Map<String, Value>> {
        self.metadata.as_ref()
    }

    // -- mutation (writer support) ----------------------------------------- //

    pub fn insert(&mut self, name: String, info: TensorInfo) {
        if let Some(slot) = self.tensors.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = info;
        } else {
            self.tensors.push((name, info));
        }
    }

    pub fn set_metadata(&mut self, meta: Map<String, Value>) {
        self.metadata = Some(meta);
    }

    /// Validate that all offsets fit within `data_len` and sizes match
    /// shape×dtype where computable.
    pub fn validate(&self, path: &Path, data_len: u64) -> Result<()> {
        for (_name, info) in &self.tensors {
            let (start, end) = info.data_offsets;
            if end > data_len || start > end {
                return Err(Error::OffsetsOutOfRange {
                    path: path.to_path_buf(),
                    start,
                    end,
                    data_len,
                });
            }
            if end - start == 0 && info.shape.iter().product::<u64>() == 0 {
                continue; // zero-sized tensor is fine
            }
            if let Some(expected) = Self::expected_size(info) {
                if end - start != expected && expected != 0 {
                    return Err(Error::SizeMismatch {
                        path: path.to_path_buf(),
                        len: end - start,
                        expected,
                    });
                }
            }
        }
        Ok(())
    }
}
