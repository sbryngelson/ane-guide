#!/usr/bin/env bash
# Build the guide into a single PDF (build/ane-guide.pdf).
# Requires pandoc and a LaTeX engine (xelatex). Chapters are taken in SUMMARY.md
# order, grouped under Part divider pages.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

# Ordered file list from the live links in SUMMARY.md (HTML comment stripped).
perl -0777 -pe 's/<!--.*?-->//gs' SUMMARY.md \
  | grep -oE '\]\(([^)]+\.md)\)' | sed -E 's/\]\(|\)//g' > build/order.txt

# Generate the Part divider pages into build/. Each divider carries an eyebrow
# part label, the title, and the list of chapters in the part (their numbers and
# titles, pulled from SUMMARY.md), rendered by the \anepartopen / \anechap /
# \anepartclose macros in pandoc/header.tex.
declare -A part_title=(
  [part-1-machine]="Part I. The Machine"
  [part-2-reaching]="Part II. Reaching the ANE"
  [part-3-performance]="Part III. Performance and Fit"
  [part-4-workloads]="Part IV. Workloads"
  [part-5-practice]="Part V. Practice"
  [part-6-silicon]="Part VI. The Silicon"
  [part-7-toolchain]="Part VII. The Toolchain and Encoding"
  [part-8-system-internals]="Part VIII. System Internals"
  [part-9-cross-silicon]="Part IX. Cross-Silicon Reference"
  [back-matter]="Back Matter"
  [appendices]="Appendices")
# One-line description per chapter, keyed by the chapter title as it appears on
# the divider (the text after the number). Rendered in grey under each row by
# \anechap so the divider reads as a "what is in this part" overview.
declare -A blurb=(
  ["What the ANE is"]="The fixed-function fp16 datapath, its wide accumulator, and the route below Core ML."
  ["Execution model"]="Compile once and dispatch many; the per-call latency budget and the program cache."
  ["Numerics"]="fp16 end to end, the wide accumulator, saturation, and where precision breaks down."
  ["Capability surface"]="Which operations the engine runs, against which a capability bit only advertises."
  ["Software stack"]="The layers from a graph to the engine, and the daemon that brokers access."
  ["Dispatching without Core ML"]="The five-step direct route: build, compile, load, bind, and dispatch."
  ["Weights and compression"]="The int4, sparse, int8, and blockwise forms, and which stream natively."
  ["Entitlement boundary"]="The signed-load gate and the entitlements that bound the reachable surface."
  ["Roofline"]="Two ceilings, the ridge point, the 2 MB working set, and the dispatch floor."
  ["Power and efficiency"]="Engine draw, energy per FLOP, and the efficiency margin over the GPU."
  ["ANE, GPU, and CPU"]="Which processor leads on each workload class, by speed and by energy."
  ["Across the chip family"]="How the roofline scales from the M1 to the M5, predicted then measured."
  ["Vision, convolution, and encoders"]="The engine's strongest classes, fitted as one fused program."
  ["LLM case study"]="Why decode belongs on the GPU, and the hybrid placement that remains."
  ["Training on the engine"]="Gradients as graph inputs, resident optimizer state, and cross-generation parity."
  ["Numerical and scientific computing"]="Stencils, Fourier transforms, and other non-network work cast as matmuls."
  ["Model-design rules"]="The validator limits, the working-set rule, and the alignment that keeps a layer fast."
  ["Optimization and the cost model"]="The three-stage latency estimate and the autotuner that ranks rewrites."
  ["Pitfalls and limits"]="The silent, target-specific failure modes and how to avoid each."
  ["Interlude. Below the API: how the engine works"]="The transition from the programming surface to the engine beneath it."
  ["Datapath and MAC geometry"]="The multiply array, the output-channel groups, and the per-core tiling."
  ["Memory hierarchy"]="The on-chip working set, the 2 MB threshold, and the streaming boundary."
  ["Compiler"]="The frontend, the anec backend dialect, lowering, and the validators."
  ["Program and container format"]="The on-disk program, its descriptors, and the resolved layout sidecar."
  ["HAL and capability gates"]="The per-chip parameter table and the capability bytes that gate each operation."
  ["Compression internals"]="The reconstruction codecs and the wire layout of each compressed weight form."
  ["Hidden layers and direct netplist authoring"]="Native layers reachable only by authoring the netplist directly."
  ["Kernel driver and IOKit ABI"]="The user clients, the selectors, and the IOKit ABI of the coprocessor endpoint."
  ["Address translation and the DART"]="The engine IOMMU, the leaf page-table entry, and the firmware rebase."
  ["Firmware"]="The real-time controller, its task model, and the dispatch loop."
  ["Host-to-firmware command protocol"]="The ninety-three-command mailbox protocol across the host boundary."
  ["Power and thermal"]="Idle and active draw, the credit sequence, and sustained thermal behavior."
  ["Security and isolation"]="The signed-load chain, secure mode, and the exclave path on the M1."
  ["Telemetry and hardware counters"]="The counter block, the stats-mask gate, and the free-running timestamp."
  ["Cross-silicon targets"]="The full target set and the rule mapping each M-series part to its H-series identity."
  ["Per-family code generation"]="How one compiler binary builds every chip from a per-family data table."
  ["Predicted upper tier"]="fp8 and the double-rate path the newest generations add."
  ["Methodology"]="The measured silicon, the tools, and what each technique can and cannot observe."
  ["Open questions"]="The findings that remain unconfirmed and what would settle each."
  ["Statements"]="Provenance, reproduction, and the work's declarations."
  ["Operation-by-device matrix"]="Every operation against every device, native or decomposed."
  ["Hidden-layer catalog"]="The native layers and their descriptors, with parameters and symbols."
  ["Decoded reference tables"]="The enum tokens, structs, status codes, register map, and program schema."
  ["Glossary"]="The terms, the family and silicon map, and the core facts to read first."
  ["References"]="The cited external literature and the Apple documentation pages."
  ["Provenance"]="The evidentiary basis of every claim, Part by Part and chapter by chapter.")
declare -A part_of=()
summary_clean=$(perl -0777 -pe 's/<!--.*?-->//gs' SUMMARY.md)
for d in "${!part_title[@]}"; do
  full="${part_title[$d]}"
  if [[ "$full" == *". "* ]]; then eyebrow="${full%%. *}"; title="${full#*. }"; else eyebrow=""; title="$full"; fi
  {
    printf '```{=latex}\n'
    printf '\\anepartopen{%s}{%s}\n' "$eyebrow" "$title"
    printf '%s\n' "$summary_clean" \
      | grep -oE "\- \[[^]]+\]\($d/[^)]+\.md\)" \
      | sed -E 's/^- \[([^]]+)\]\(.*/\1/' \
      | while IFS= read -r text; do
          text="${text//&/\\&}"; text="${text//%/\\%}"
          num=""; title="$text"
          if [[ "$text" =~ ^([0-9]+)\.\ (.*)$ ]]; then
            num="${BASH_REMATCH[1]}"; title="${BASH_REMATCH[2]}"
          elif [[ "$text" =~ ^Appendix\ ([A-Z])\.\ (.*)$ ]]; then
            num="${BASH_REMATCH[1]}"; title="${BASH_REMATCH[2]}"
          fi
          b="${blurb[$title]:-}"; b="${b//&/\\&}"; b="${b//%/\\%}"
          printf '\\anechap{%s}{%s}{%s}\n' "$num" "$title" "$b"
        done
    printf '\\anepartclose\n'
    # Back matter is unnumbered (\chapter*), so its floats would otherwise keep
    # the last numbered chapter's counter (Figure 36.1, Table 36.10, ...). Drop
    # the chapter prefix and restart the float counters for the back matter.
    [ "$d" = "back-matter" ] && printf '\\setcounter{figure}{0}\\setcounter{table}{0}\\renewcommand{\\thefigure}{\\arabic{figure}}\\renewcommand{\\thetable}{\\arabic{table}}\n'
    # Appendices number as A, B, ...; switch LaTeX into appendix mode after the
    # divider page, and restore the chapter-prefixed float numbering (now A.1, ...).
    [ "$d" = "appendices" ] && printf '\\appendix\\ANEinappendixtrue\\renewcommand{\\thefigure}{\\thechapter.\\arabic{figure}}\\renewcommand{\\thetable}{\\thechapter.\\arabic{table}}\n'
    printf '```\n'
  } > "build/divider-$d.md"
  part_of[$d]="build/divider-$d.md"
done

# Assemble the input list, inserting a Part divider when the group changes.
assembled=(); prev=""
while IFS= read -r f; do
  dir="${f%%/*}"
  if [ "$dir" = "$f" ]; then          # root-level file (no Part divider)
    assembled+=( "$f" ); prev="root"; continue
  fi
  if [ "$dir" != "$prev" ]; then
    assembled+=( "${part_of[$dir]}" ); prev="$dir"
  fi
  assembled+=( "$f" )
done < build/order.txt

ABSTRACT="The Apple Neural Engine (ANE) is the fixed-function matrix accelerator that has shipped in Apple systems-on-chip since the A11-class iPhone and iPad chips and the M1-class Mac chips, exposed to applications only through the Core ML model framework. This guide reports a reverse-engineered account of the engine, based on direct measurement on Apple silicon and static analysis of the private runtime, compiler, kernel driver, and firmware. It documents the datapath and the roofline that bound the engine's throughput and energy, the dispatch route that reaches it below Core ML, the compiler and on-disk program format, the weight-compression scheme, and the kernel driver, firmware, and command protocol beneath them. The account covers the A11 through A18 and M1 through M5 families, with per-chip target tables and an operation-by-device matrix; the direct measurements are on the M1 and M5. Claims are labeled as measured, decompile-derived, or predicted, and the methodology and open questions are recorded. The direct route is callable from ordinary user space but remains undocumented, unsupported, and version-fragile; it is intended for measurement, research, and on-device work, not for shipping software, where Core ML remains the supported path."

pandoc introduction.md "${assembled[@]}" \
  -o build/ane-guide.tex \
  --standalone \
  -H pandoc/header.tex \
  --syntax-definition=pandoc/mlir.xml \
  --syntax-definition=pandoc/fbs.xml \
  --highlight-style=pandoc/ane.theme \
  --number-sections \
  -V secnumdepth=2 \
  --lua-filter=pandoc/numbering.lua \
  --lua-filter=pandoc/mermaid.lua \
  --lua-filter=pandoc/listing.lua \
  --lua-filter=pandoc/callout.lua \
  --lua-filter=pandoc/crossref.lua \
  --lua-filter=pandoc/xref.lua \
  --lua-filter=pandoc/citations.lua \
  --toc --toc-depth=1 \
  --top-level-division=chapter \
  -V geometry:margin=1in \
  -V fontsize=10pt \
  -V documentclass=report \
  -V colorlinks=true \
  -M title="Apple Neural Engine" \
  -M subtitle="Architecture, Programming, and Performance" \
  -M author="Spencer H. Bryngelson" \
  -M date="June 2026" \
  -M abstract="$ABSTRACT"

# Stamp the title page with the revision date (the last commit's date), read by
# the \anebuilddate macro in pandoc/header.tex. Falls back to the macro default
# when not in a git checkout.
build_date=$(git log -1 --format=%cd --date=format:'%B %-d, %Y' 2>/dev/null || true)
[ -n "$build_date" ] && printf '\\def\\anebuilddate{%s}\n' "$build_date" > build/anedate.tex || rm -f build/anedate.tex

# Run XeLaTeX three times (inside build/, so aux reads/writes and the
# diagrams/*.pdf image paths all resolve) to settle the TOC, the list of figures
# and tables, and any cross-references.
( cd build && for i in 1 2 3; do
    xelatex -interaction=nonstopmode ane-guide.tex >/dev/null 2>&1 || true
  done )
[ -f build/ane-guide.pdf ] && echo "built build/ane-guide.pdf (${#assembled[@]} inputs)" || { echo "BUILD FAILED"; tail -20 build/ane-guide.log; exit 1; }
