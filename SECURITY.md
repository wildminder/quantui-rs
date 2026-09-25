# Security policy

## Supported versions

`quantui-rs` is pre-1.0. Security fixes are applied to the latest `main`
branch and the most recent tagged release.

## Reporting a vulnerability

Do not open a public issue for a security vulnerability. Use the repository's
private GitHub vulnerability reporting form:

https://github.com/wildminder/quantui-rs/security/advisories/new

Include the affected version or commit, the input format involved
(`.safetensors` or `.gguf`), a minimal reproduction if possible, and the
observed impact.

Because the tool parses untrusted binary model files, malformed-file crashes,
out-of-bounds reads, excessive memory allocation, and output-path issues are in
scope. Please give the maintainers a reasonable opportunity to ship a fix
before public disclosure.
