# Web edition

This builds the guide into a browsable static website with
[mdBook](https://rust-lang.github.io/mdBook/), from the same chapter sources the
PDF pipeline (`../build.sh`) uses. The two outputs read the same markdown and
stay in sync; nothing here modifies the authoring sources.

## Build

```bash
./build-web.sh
open book/index.html
```

The script regenerates `src/` from the chapters, then runs mdBook. To preview
with live reload instead, run `.tools/mdbook serve` after a first build.

## How it works

`gen.py` copies `../part-*`, `../back-matter`, and `../appendices` into `src/`
and translates the constructs the PDF pipeline expects into mdBook-native form:

- `(ex:name)` demo links become repository blob URLs (the map mirrors
  `../pandoc/examples.lua`).
- Inline `[Key]` citations become links into the references appendix (mirrors
  `../pandoc/citations.lua`), whose key cells carry matching HTML anchors.
- ` ```mermaid ` blocks become `<pre class="mermaid">` figures, rendered in the
  browser; a `%% caption:` line becomes the figure caption.
- Pandoc attribute syntax (`{.unnumbered}`, `{#tbl:...}`) is stripped, since
  CommonMark has none.

Math (`$...$`, `$$...$$`) is left for the `mdbook-katex` preprocessor, which
extracts it before markdown so subscripts survive.

## Toolchain

- A project-local mdBook 0.4.48 in `.tools/`, built on first run. mdBook is
  pinned to the 0.4 line because the math preprocessor `mdbook-katex` (latest,
  0.9.4) speaks the 0.4 preprocessor protocol and is not compatible with mdBook
  0.5. With a Rust toolchain present `build-web.sh` compiles it native with
  `cargo`; without Rust it falls back to the prebuilt x86_64 binary (Rosetta on
  Apple silicon).
- `mdbook-katex` on `PATH` (install with `cargo install mdbook-katex`).
- The KaTeX stylesheet and `mermaid.min.js` load from a CDN, so a first build and
  a first view both want network access; later builds are offline.

Everything under `src/`, `book/`, `.tools/`, and `mermaid.min.js` is generated or
vendored and is git-ignored.
