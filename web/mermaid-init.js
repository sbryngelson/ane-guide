// Render the <pre class="mermaid"> blocks that gen.py emits. Loaded after
// mermaid.min.js (see book.toml).
//
// The guide's web edition is locked to the dark theme, so this is a dark mermaid
// palette: dark node fills, light text and edges, transparent figure background
// (no white card). The PDF pipeline (pandoc/mermaid-config.json) keeps the light
// steel-blue palette because its pages are white, so the two intentionally
// differ; gen.py remaps the per-node style fills (the five semantic colours) to
// the dark variants for the web build.
mermaid.initialize({
  startOnLoad: true,
  theme: "base",
  themeVariables: {
    fontFamily: "Source Sans Pro, Helvetica Neue, Arial, sans-serif",
    fontSize: "13px",
    primaryColor: "#22323f",
    primaryBorderColor: "#6fa8c9",
    primaryTextColor: "#e7e6df",
    secondaryColor: "#2a2f34",
    tertiaryColor: "#1c2024",
    lineColor: "#8a98a6",
    textColor: "#e7e6df",
    clusterBkg: "#191e22",
    clusterBorder: "#3a434c",
    titleColor: "#cfd6cd",
    edgeLabelBackground: "#1d1f21",
    nodeBorder: "#6fa8c9",
    actorBkg: "#22323f",
    actorBorder: "#6fa8c9",
    actorTextColor: "#e7e6df",
    actorLineColor: "#6b7681",
    signalColor: "#8a98a6",
    signalTextColor: "#e7e6df",
    labelBoxBkgColor: "#22323f",
    labelBoxBorderColor: "#6fa8c9",
    labelTextColor: "#e7e6df",
    loopTextColor: "#e7e6df",
    noteBkgColor: "#342d18",
    noteBorderColor: "#c9a85a",
    noteTextColor: "#e7e6df",
  },
  flowchart: { useMaxWidth: true, nodeSpacing: 30, rankSpacing: 34, padding: 4 },
  sequence: {
    useMaxWidth: true,
    width: 120,
    actorMargin: 32,
    diagramMarginX: 8,
    diagramMarginY: 8,
    messageMargin: 24,
    actorFontFamily: "Source Sans Pro, Helvetica Neue, Arial, sans-serif",
    noteFontFamily: "Source Sans Pro, Helvetica Neue, Arial, sans-serif",
    messageFontFamily: "Source Sans Pro, Helvetica Neue, Arial, sans-serif",
  },
});
