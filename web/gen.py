#!/usr/bin/env python3
"""Generate an mdBook source tree (web/src/) from the guide's authoring sources.

The chapter markdown under ../part-*, ../back-matter and ../appendices is written
for the pandoc -> PDF pipeline (build.sh). This script copies that markdown into
web/src/ and rewrites the handful of pandoc-specific constructs into mdBook-native
equivalents, then emits an mdBook SUMMARY.md from the authoring ../SUMMARY.md.

The authoring sources are never modified. Everything under web/src/ is generated
output and is safe to delete; this script recreates it. The PDF pipeline and this
web pipeline read the same chapter files and stay in sync.

What gets translated:
  - inline [Key] citations          -> links into appendix E   (mirrors citations.lua)
  - "chapter N" / "Appendix X"      -> links to that chapter   (mirrors xref.lua)
  - [text]{#ref-key} span anchors   -> HTML anchors            (appendix E targets)
  - {.unnumbered} / {#tbl:..} attrs -> stripped (CommonMark has no attribute syntax)
Math ($...$, $$...$$) and ```mermaid blocks are left verbatim for the
mdbook-katex and mdbook-mermaid preprocessors to handle.
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
GUIDE = os.path.dirname(HERE)
SRC = os.path.join(HERE, "src")

# Directories of authoring content to publish, in book order. Anything outside
# this list (pandoc/, build/) is excluded.
CONTENT_DIRS = [
    "part-1-machine", "part-2-reaching", "part-3-performance", "part-4-workloads",
    "part-5-practice", "part-6-silicon", "part-7-toolchain",
    "part-8-system-internals", "part-9-cross-silicon", "back-matter", "appendices",
]

# --- citations.lua: inline [Key] -> link into appendix E.0 ------------------
CITE_KEYS = ["AppleCoreMLTools", "AppleCoreML", "AppleANE", "AppleAccelerate",
             "AppleVision", "AppleActiveDevices2026", "Williams2009", "Orion2026",
             # E.1 external literature, cited in the Introduction's Related work
             "Choi2013", "Ilic2014", "Ding2019", "Yang2020", "Verhelst2025",
             "Jouppi2017", "Ignatov2019", "Jayanth2024", "Prashanthi2025",
             "Fanariotis2025", "Tummalapalli2026", "Bi2026", "Xu2025", "Moon2025",
             "Chen2025", "Benazir2026", "Hubner2025", "tinygrad", "eilnANE",
             "Singh2026", "libane", "whispercpp", "Hollemans",
             "AppleANETransformers", "Plyenkov2019", "CommunityANE", "Zeus2025",
             "ANEForge2026"]

CITE_RE = re.compile(r"\[(" + "|".join(sorted(CITE_KEYS, key=len, reverse=True)) + r")\]")
SPAN_RE = re.compile(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\{#(ref-[a-z0-9]+)\}")
ATTR_RE = re.compile(r"\s*\{(?:\.unnumbered|#(?:tbl|fig|lst):[a-z0-9-]+)\}")
# xref.lua mirror: in-text "chapter N" / "Appendix X" -> a link to that chapter.
XREF_RE = re.compile(r"\b([Cc]hapters?|[Aa]ppendix)[ ]+(\d+|[A-Z])\b")

# --- Float numbering (Table / Listing / Figure N.M), mirroring the PDF cleveref.
# The PDF numbers floats per chapter via newfloat + cleveref; mdBook does not
# number floats at all. scan_floats() assigns the same N.M numbers here, captions
# carry them, and [text](#tbl:slug) references render as "Table N.M" with a live
# anchor, so a cross-reference reads identically in both editions.
FLOAT_LABEL_RE = re.compile(r"\{#((?:tbl|fig|lst):[A-Za-z0-9:._-]+)\}")
FLOATREF_RE = re.compile(r"\[([^\]]*)\]\(#((?:tbl|fig|lst):[A-Za-z0-9:._-]+)\)")
_TYPE = {"tbl": "Table", "fig": "Figure", "lst": "Listing"}


def _slug_anchor(slug):
    return slug.replace(":", "-")


def _ordered_files():
    """Content files in the same order build_src() writes them."""
    files = []
    for d in CONTENT_DIRS:
        for root, _, fs in os.walk(os.path.join(GUIDE, d)):
            for fn in sorted(fs):
                if fn.endswith(".md"):
                    files.append(os.path.relpath(os.path.join(root, fn), GUIDE).replace(os.sep, "/"))
    return files


def scan_floats(chap, app):
    """Pre-scan all content in book order and assign each captioned table, listing,
    and figure a per-chapter number N.M (M restarts per chapter, per type), matching
    the PDF. An unnumbered back-matter chapter keeps the previous chapter context,
    as a LaTeX \\chapter* does. Returns (labels, file_floats):
    labels[slug] = (display, relpath, anchor); file_floats[relpath] is the ordered
    list of (type, 'N.M', anchor_or_None) the captions in that file take."""
    p2num = {v: k for k, v in chap.items()}
    p2app = {v: k for k, v in app.items()}
    labels, file_floats = {}, {}
    cur, counters = None, {}

    def nextnum(t):
        counters[(cur, t)] = counters.get((cur, t), 0) + 1
        return f"{cur}.{counters[(cur, t)]}"

    def record(t, slug, rel):
        num = nextnum(t)
        anc = _slug_anchor(slug) if slug else None
        if slug:
            labels[slug] = (f"{_TYPE[t]} {num}", rel, anc)
        return (t, num, anc)

    for rel in _ordered_files():
        if rel in p2num:
            cur = p2num[rel]
        elif rel in p2app:
            cur = p2app[rel]
        elif rel == "part-5-practice/handoff.md":
            cur = "I"        # the Interlude numbers its float I.1, matching the PDF
        flist = []
        lines = open(os.path.join(GUIDE, rel), encoding="utf-8").read().split("\n")
        i, in_fence = 0, False
        while i < len(lines):
            st = lines[i].lstrip()
            if st.startswith("```mermaid"):
                i += 1
                while i < len(lines) and not lines[i].lstrip().startswith("```"):
                    cap = re.match(r"\s*%%\s*caption:\s*(.*)$", lines[i])
                    if cap and cur:
                        m = FLOAT_LABEL_RE.search(cap.group(1))
                        flist.append(record("fig", m.group(1) if m else None, rel))
                    i += 1
                i += 1
                continue
            if st.startswith("```") or st.startswith("~~~"):
                in_fence = not in_fence
                i += 1
                continue
            if not in_fence and cur:
                m = re.match(r"(Table|Listing):\s", lines[i])
                if m:
                    t = "tbl" if m.group(1) == "Table" else "lst"
                    lab = FLOAT_LABEL_RE.search(lines[i])
                    flist.append(record(t, lab.group(1) if lab else None, rel))
            i += 1
        file_floats[rel] = flist
    return labels, file_floats

# The web edition's mermaid theme is dark, so remap the five semantic figure
# colours to dark variants in the per-node `style fill:`
# directives. The PDF keeps the light originals (its pages are white).
MERMAID_DARK = {
    "#dce7f2": "#22323f", "#1f4e79": "#6fa8c9",   # primary / highlight
    "#e7ebef": "#2a2f34", "#5b6b7b": "#8a98a6",   # inert / memory
    "#deebe1": "#1e3327", "#3f7a4e": "#5fb38a",   # good / compute-bound
    "#f5ecd6": "#342d18", "#9c7b34": "#c9a85a",   # caution / bandwidth-bound
    "#f1dade": "#3a201e", "#a8332f": "#d9756f",   # negative / rejected
}
MERMAID_HEX_RE = re.compile("|".join(re.escape(k) for k in MERMAID_DARK), re.IGNORECASE)


def summary_targets():
    """Map a chapter number ('9') and an appendix letter ('C') to its authoring
    path, read from ../SUMMARY.md. Lets the web build linkify in-text 'chapter N'
    and 'Appendix X' references the way pandoc/xref.lua does for the PDF."""
    chap, app = {}, {}
    with open(os.path.join(GUIDE, "SUMMARY.md"), encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"\s*- \[0*(\d+)\.[^\]]*\]\(([^)]+\.md)\)", line)
            if m:
                chap[m.group(1)] = m.group(2)
                continue
            m = re.match(r"\s*- \[Appendix ([A-Z])\.[^\]]*\]\(([^)]+\.md)\)", line)
            if m:
                app[m.group(1)] = m.group(2)
    return chap, app


def rewrite(text, relpath, chap, app, labels=None, floats=None):
    """Translate one chapter's pandoc constructs into mdBook-native markdown."""
    is_references = relpath == "references.md"
    base = os.path.dirname(relpath)
    labels = labels or {}
    floats = list(floats or [])
    _fi = [0]

    def take():
        if _fi[0] < len(floats):
            e = floats[_fi[0]]
            _fi[0] += 1
            return e
        return (None, "?", None)

    def float_ref_sub(m):
        text, slug = m.group(1), m.group(2)
        info = labels.get(slug)
        if not info:
            return m.group(0)
        display, deffile, anchor = info
        first = next((c for c in text if c.isalpha()), "")
        if first and first.islower():       # mid-sentence: lowercase, matching \cref
            display = display[0].lower() + display[1:]
        href = f"#{anchor}" if deffile == relpath else \
            os.path.relpath(deffile, base).replace(os.sep, "/") + f"#{anchor}"
        return f"[{display}]({href})"

    def xref_sub(m):
        word, key = m.group(1), m.group(2)
        target = app.get(key) if word.lower() == "appendix" else chap.get(key)
        if not target:
            return m.group(0)
        rel = os.path.relpath(target, base).replace(os.sep, "/")
        return f"{word} [{key}]({rel})"

    def cite_sub(m):
        key = m.group(1)
        tgt = os.path.relpath("references.md", base).replace(os.sep, "/")
        return f"[\\[{key}\\]]({tgt}#ref-{key.lower()})"

    lines = text.splitlines()
    out, i, in_fence = [], 0, False
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # A ```mermaid block becomes a <pre class="mermaid"> element that the
        # client-side mermaid.js renders. The body is emitted verbatim; CommonMark
        # treats <pre> as a raw-HTML block and does not reinterpret the diagram.
        if stripped.startswith("```mermaid"):
            body, caption = [], None
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                cap = re.match(r"\s*%%\s*caption:\s*(.*?)\s*$", lines[i])
                if cap:  # the PDF pipeline's figure caption, carried as a mermaid comment
                    caption = ATTR_RE.sub("", cap.group(1)).strip()
                else:
                    body.append(MERMAID_HEX_RE.sub(
                        lambda m: MERMAID_DARK[m.group(0).lower()], lines[i]))
                i += 1
            i += 1  # skip the closing fence
            out.append("")  # blank line so CommonMark starts a raw-HTML block
            fnum = ""
            if caption:
                _t, fnum, fanc = take()
                idattr = f' id="{fanc}"' if fanc else ""
                out.append(f"<figure{idattr}>")
            out.append('<pre class="mermaid">')
            out.extend(body)
            out.append("</pre>")
            if caption:
                cap_txt = FLOATREF_RE.sub(float_ref_sub, CITE_RE.sub(cite_sub, caption))
                out.append(f"<figcaption><strong>Figure {fnum}.</strong> {cap_txt}</figcaption>")
                out.append("</figure>")
            out.append("")
            continue
        # Pandoc raw blocks for the PDF (```{=latex} / ```{=tex}) are not web
        # content; drop the whole block so the LaTeX never reaches the site.
        if re.match(r"(```|~~~)\{=(latex|tex)\}", stripped):
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(("```", "~~~")):
                i += 1
            i += 1  # skip the closing fence
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue
        # Appendix E carries the citation anchors and the key definitions; turn the
        # span anchors into HTML targets and do not linkify the keys to themselves.
        if is_references:
            line = SPAN_RE.sub(r'<a id="\2"></a>\1', line)
        else:
            line = CITE_RE.sub(cite_sub, line)
        line = ATTR_RE.sub("", line)
        if not stripped.startswith("#"):     # don't linkify a chapter/appendix title to itself
            line = XREF_RE.sub(xref_sub, line)
        line = FLOATREF_RE.sub(float_ref_sub, line)
        # Number a Table:/Listing: caption (matching the PDF) and anchor a labelled one.
        cm = re.match(r"(Table|Listing):\s+(.*)$", line)
        if cm:
            _t, num, anc = take()
            idtag = f'<a id="{anc}"></a>' if anc else ""
            line = f"{idtag}**{cm.group(1)} {num}.** {cm.group(2)}"
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def build_src():
    if os.path.isdir(SRC):
        shutil.rmtree(SRC)
    os.makedirs(SRC)
    chap, app = summary_targets()
    labels, file_floats = scan_floats(chap, app)
    for d in CONTENT_DIRS:
        srcdir = os.path.join(GUIDE, d)
        for root, _, files in os.walk(srcdir):
            for fn in sorted(files):
                if not fn.endswith(".md"):
                    continue
                abs_in = os.path.join(root, fn)
                rel = os.path.relpath(abs_in, GUIDE)
                relkey = rel.replace(os.sep, "/")
                abs_out = os.path.join(SRC, rel)
                os.makedirs(os.path.dirname(abs_out), exist_ok=True)
                with open(abs_in, encoding="utf-8") as fh:
                    text = fh.read()
                with open(abs_out, "w", encoding="utf-8") as fh:
                    fh.write(rewrite(text, relkey, chap, app, labels,
                                     file_floats.get(relkey, [])))

    # References is a standalone top-level section, not an appendix, so it lives at
    # the repo root rather than in a CONTENT_DIRS directory. Process it like any
    # content file so its [Key] anchors and formatting match.
    refs = os.path.join(GUIDE, "references.md")
    if os.path.exists(refs):
        with open(refs, encoding="utf-8") as fh:
            text = fh.read()
        with open(os.path.join(SRC, "references.md"), "w", encoding="utf-8") as fh:
            fh.write(rewrite(text, "references.md", chap, app, labels,
                             file_floats.get("references.md", [])))

    # Stage the built PDF at the book root so the web "Download PDF" button
    # (home-link.js links to /apple-neural-engine-guide.pdf) resolves. The PDF is
    # produced by build.sh at build/ane-guide.pdf; mdBook copies non-markdown
    # files from src/ into the served book/. Copy it in when present.
    pdf = os.path.join(GUIDE, "build", "ane-guide.pdf")
    if os.path.exists(pdf):
        shutil.copy2(pdf, os.path.join(SRC, "apple-neural-engine-guide.pdf"))


def build_summary():
    """Emit web/src/SUMMARY.md in mdBook form from the authoring ../SUMMARY.md.

    Parts I-IX keep their real chapter number in the label ('09.' -> '9.');
    mdBook's own auto-number is hidden via custom.css so the sidebar matches the
    chapter headings and the 'chapter N' cross-reference links. Back matter and
    appendices become unnumbered suffix chapters, which keeps the 'Appendix A'
    lettering.
    """
    with open(os.path.join(GUIDE, "SUMMARY.md"), encoding="utf-8") as fh:
        text = re.sub(r"<!--.*?-->", "", fh.read(), flags=re.DOTALL)

    item = re.compile(r"^- \[([^]]+)\]\(([^)]+\.md)\)\s*$")
    parts, suffix_sections, cur, sufcur = [], [], None, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):                       # a Part divider
            cur = {"title": s[4:].strip(), "items": []}
            parts.append(cur)
            sufcur = None
            continue
        if s.startswith("## "):                         # half / back-matter / appendices
            name = s[3:].strip()
            if name in ("Back matter", "Appendices", "References"):
                sufcur = {"title": name, "items": []}
                suffix_sections.append(sufcur)
                cur = None
            else:
                sufcur = None                          # 'Front/Back half' dividers carry no items
            continue
        m = item.match(s)
        if not m:
            continue
        text_, path = m.group(1), m.group(2)
        if sufcur is not None:
            sufcur["items"].append((text_, path))      # keep full label (e.g. 'Appendix A. ...')
        elif cur is not None:
            # Keep the real chapter number ('09. ' -> '9. ') and hide mdBook's
            # own auto-number via CSS, so the sidebar matches the chapter
            # headings and the 'chapter N' cross-reference links.
            clean = re.sub(r"^0+(\d)", r"\1", text_)
            cur["items"].append((clean, path))

    lines = ["# Summary", "", "[Introduction](intro.md)", ""]
    for p in parts:
        lines.append(f"# {p['title']}")
        lines.append("")
        for txt, path in p["items"]:
            lines.append(f"- [{txt}]({path})")
        lines.append("")
    # Back matter and appendices stay unnumbered suffix chapters (so the
    # 'Appendix A' lettering is not doubled by an mdBook number; mdBook does not
    # accept a part title above suffix chapters). home-link.js inserts the
    # "Back matter" and "Appendices" section headers into the sidebar at runtime.
    for sec in suffix_sections:
        for txt, path in sec["items"]:
            lines.append(f"[{txt}]({path})")
    lines.append("")

    with open(os.path.join(SRC, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_intro():
    """Emit web/src/intro.md as the PDF's introduction.md rendered with web-relative
    citation and cross-reference links, wrapped in the web edition's own title block
    and license footer. introduction.md is the single source, so the two editions
    never drift."""
    chap, app = summary_targets()
    labels, _ = scan_floats(chap, app)
    with open(os.path.join(GUIDE, "introduction.md"), encoding="utf-8") as fh:
        body = fh.read()
    # Drop the "# Introduction" heading; the page already carries the book-title H1.
    body = re.sub(r"^#\s+Introduction\b.*\n", "", body, count=1).lstrip("\n")
    body = rewrite(body, "intro.md", chap, app, labels, [])

    header = (
        "# Apple Neural Engine\n\n"
        "## Architecture, Programming, and Performance\n\n"
        "Spencer H. Bryngelson\n\n"
        "*School of Computational Science & Engineering, "
        "Georgia Institute of Technology, Atlanta, GA 30332, USA*\n\n"
        "ORCID 0000-0003-1750-7265\n\n"
        "This is the web edition of the guide. Use the sidebar to move between the "
        "parts and chapters; the search box indexes the full text.\n\n"
    )
    footer = (
        "\n## License and citation\n\n"
        "This guide is licensed under "
        "[Creative Commons Attribution 4.0 International (CC BY 4.0)]"
        "(https://creativecommons.org/licenses/by/4.0/). Apple, Apple silicon, "
        "Core ML, and the Apple Neural Engine are trademarks of Apple Inc.; this is "
        "an independent work and is not affiliated with, authorized by, or endorsed "
        "by Apple Inc.\n\n"
        "To cite: Spencer H. Bryngelson. *Apple Neural Engine: Architecture, "
        "Programming, and Performance.* "
        "[arXiv:2606.22283](https://arxiv.org/abs/2606.22283), 2026.\n"
    )
    with open(os.path.join(SRC, "intro.md"), "w", encoding="utf-8") as fh:
        fh.write(header + body + footer)


if __name__ == "__main__":
    build_src()
    write_intro()
    build_summary()
    count = sum(1 for _, _, fs in os.walk(SRC) for f in fs if f.endswith(".md"))
    print(f"generated web/src/ ({count} markdown files)")
