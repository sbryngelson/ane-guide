#!/usr/bin/env bash
# new-edition.sh -- cut a new numbered edition of the guide.
#
# Bumps the VERSION file and prepends a dated entry to the revision history
# (back-matter/revision-history.md). It does NOT commit: the changelog prose is
# the one part worth a human eye, so review the entry, then commit and, if you
# like, tag the edition. The title-page edition number and date are stamped at
# build time from VERSION and the last commit, so the prose is the only manual
# step.
#
# Usage:
#   ./new-edition.sh "first note" "second note" ...   # seed one bullet per note
#   ./new-edition.sh                                   # placeholder bullet to fill in
set -euo pipefail
cd "$(dirname "$0")"

CHANGELOG="back-matter/revision-history.md"
cur=$(tr -d ' \n' < VERSION)
next=$((cur + 1))
# Space-padded day collapsed to no leading zero, matching the title-page format
# ("June 4, 2026", not "June 04, 2026"); portable across BSD and GNU date.
today=$(date '+%B %e, %Y' | tr -s ' ')

# One bullet per argument; a single placeholder when none are given.
notes=()
if [ "$#" -gt 0 ]; then
  for n in "$@"; do notes+=("- $n"); done
else
  notes+=("- TODO: describe the change")
fi

# Assemble the new entry, then insert it before the first existing "## v" entry
# so the page intro stays on top and editions read newest first.
entry="## v$next ($today)"$'\n\n'
for line in "${notes[@]}"; do entry+="$line"$'\n'; done

tmp=$(mktemp)
inserted=0
while IFS= read -r line || [ -n "$line" ]; do
  if [ "$inserted" -eq 0 ] && [[ "$line" == "## v"* ]]; then
    printf '%s\n' "$entry" >> "$tmp"
    inserted=1
  fi
  printf '%s\n' "$line" >> "$tmp"
done < "$CHANGELOG"
if [ "$inserted" -ne 1 ]; then
  echo "error: no '## v' entry to insert before in $CHANGELOG" >&2
  rm -f "$tmp"; exit 1
fi
mv "$tmp" "$CHANGELOG"
printf '%s\n' "$next" > VERSION

echo "Bumped VERSION to $next; added a v$next ($today) entry to $CHANGELOG."
echo
if git rev-parse -q --verify "refs/tags/guide-v$cur" >/dev/null 2>&1; then
  echo "Commits since guide-v$cur (crib for the changelog bullets):"
  git log --format='  %s' "guide-v$cur..HEAD"
else
  echo "Recent commits (crib for the changelog bullets):"
  git log --format='  %s' -12
fi
echo
echo "Next:"
echo "  1. edit $CHANGELOG (fill in the bullets)"
echo "  2. git add VERSION $CHANGELOG && git commit"
echo "  3. git tag guide-v$next   # optional, anchors the next edition's diff range"
