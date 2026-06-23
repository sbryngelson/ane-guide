#!/usr/bin/env bash
# Build the web edition of the guide into web/book/ (open web/book/index.html).
#
# This regenerates web/src/ from the authoring chapters and runs mdBook. It does
# not touch the authoring sources or the PDF pipeline.
#
# Toolchain: a project-local mdBook 0.4.48 in .tools/ and mdbook-katex from cargo's
# bin dir. mdBook is pinned to the 0.4 line because the math preprocessor
# mdbook-katex (latest, 0.9.4) speaks the 0.4 preprocessor protocol and is not
# compatible with mdBook 0.5. With a Rust toolchain present the pinned mdBook is
# built native; otherwise it is fetched as a prebuilt x86_64 binary (Rosetta).
# Math CSS and mermaid.js load from a CDN on the first build, then cache locally.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

MDBOOK_VERSION="0.4.48"
MDBOOK=".tools/bin/mdbook"
if [ ! -x "$MDBOOK" ]; then
  if command -v cargo >/dev/null; then
    echo "building mdBook $MDBOOK_VERSION (native) ..."
    cargo install mdbook --version "$MDBOOK_VERSION" --root .tools
  else
    echo "fetching prebuilt mdBook $MDBOOK_VERSION ..."
    mkdir -p .tools/bin
    curl -fsSL "https://github.com/rust-lang/mdBook/releases/download/v${MDBOOK_VERSION}/mdbook-v${MDBOOK_VERSION}-x86_64-apple-darwin.tar.gz" \
      | tar xz -C .tools/bin mdbook
    chmod +x "$MDBOOK"
  fi
fi
if ! command -v mdbook-katex >/dev/null; then
  echo "error: mdbook-katex not found on PATH (install with: cargo install mdbook-katex)" >&2
  exit 1
fi

# 1. Regenerate the mdBook source tree from the authoring chapters.
python3 gen.py

# 2. Vendor the mermaid runtime once (the ```mermaid blocks are rendered client
#    side; gen.py emits the <pre class="mermaid"> wrappers, mermaid-init.js is
#    hand-written). Fetched only if absent, so later builds are offline. Kept on
#    the same major as the PDF pipeline's mermaid-cli@11 so layouts match.
if [ ! -f mermaid.min.js ]; then
  echo "fetching mermaid.min.js ..."
  curl -fsSL https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js -o mermaid.min.js
fi

# 3. Build the static site.
"$MDBOOK" build

echo "built web/book/ — open web/book/index.html"
