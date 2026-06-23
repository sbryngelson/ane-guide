#!/usr/bin/env bash
# check.sh -- build/structure invariants for the ANE guide. Exit non-zero on any violation.
# Editorial/style rules were intentionally dropped; only checks that keep the PDF,
# the web edition, and the GitHub/Read-the-Docs render building correctly remain:
#   1. Spine<->file consistency: every .md link target in SUMMARY.md exists, and
#      every chapter .md file is linked exactly once in SUMMARY.md.
#   2. Every relative .md link inside any chapter file resolves to a real file.
#   3. GitHub-incompatible math: chapters must not use KaTeX macros GitHub rejects
#      (\operatorname, \newcommand, ...) or the escaped star \*, so the math renders
#      on GitHub and Read the Docs as well as in the PDF.
# Non-chapter files excluded from the per-chapter rules: README.md, SUMMARY.md,
# BUILD.md, introduction.md. With zero chapter files this passes by design.
set -uo pipefail
cd "$(dirname "$0")"
fail=0
note(){ echo "CHECK FAIL: $1"; fail=1; }

# SUMMARY.md with HTML comments stripped, so the planned-ToC block (which shows
# example link syntax and bare paths) is never treated as a live link.
summary_live=$(perl -0777 -pe 's/<!--.*?-->//gs' SUMMARY.md)

# 1. SUMMARY links resolve
mapfile -t linked < <(printf '%s' "$summary_live" | grep -oE '\]\(([^)]+\.md)\)' | sed -E 's/\]\(|\)//g')
for L in "${linked[@]}"; do
  [ -f "$L" ] || note "SUMMARY links missing file: $L"
done

# 1 (cont.) + 2. per chapter file: linked exactly once, relative links resolve
while IFS= read -r f; do
  rel="${f#./}"
  case "$rel" in README.md|SUMMARY.md|BUILD.md|introduction.md) continue;; esac
  cnt=$(printf '%s' "$summary_live" | grep -cF "($rel)")
  [ "$cnt" = "1" ] || note "chapter not linked exactly once in SUMMARY ($cnt): $rel"
  # relative .md links inside this chapter must resolve (relative to the file's dir)
  dir=$(dirname "$f")
  while IFS= read -r tgt; do
    [ -z "$tgt" ] && continue
    case "$tgt" in
      http://*|https://*|/*) continue;;
    esac
    [ -f "$dir/$tgt" ] || note "broken link in $rel -> $tgt"
  done < <(grep -oE '\]\(([^)]+\.md)\)' "$f" | sed -E 's/\]\(|\)//g')
done < <(find . -name '*.md' -not -path '*/build/*' -not -path '*/web/*' -not -path '*/arxiv/*' ! -name SUMMARY.md ! -name README.md)

# 3. GitHub-incompatible math (chapters only): GitHub renders a restricted KaTeX
# subset and rejects several macros (notably \operatorname) and the escaped star
# \*. Use \mathrm{...} instead of \operatorname{...}, a literal * not \*, and no
# \def/\newcommand/custom macros. Fail on the known-disallowed constructs.
while IFS= read -r f; do
  rel="${f#./}"
  case "$rel" in README.md|SUMMARY.md|BUILD.md|introduction.md) continue;; esac
  hits=$(grep -nE '\\(operatorname|def|newcommand|gdef|edef|let|require|htmlClass|htmlId|htmlData|htmlStyle|includegraphics)\b|\\\*' "$f" || true)
  if [ -n "$hits" ]; then note "GitHub-incompatible math macro in $rel:"; echo "$hits"; fi
done < <(find . -name '*.md' -not -path '*/build/*' -not -path '*/web/*' -not -path '*/arxiv/*' ! -name SUMMARY.md ! -name README.md)

[ "$fail" = 0 ] && echo "check.sh: PASS"
exit $fail
