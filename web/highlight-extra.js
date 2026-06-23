// Highlight the listings mdBook's bundled highlight.js leaves plain: the `text`
// pseudocode (recipes), and the `mlir` and `fbs` blocks (languages it does not
// ship). mdBook highlights the C / Python / Swift blocks itself; this only
// touches the ones it skips, re-highlighting them after its own pass (window
// load, which fires after mdBook's DOMContentLoaded handler). highlight.js here
// is v10, so the call signature is hljs.highlight(language, code, ignoreIllegals).
(function () {
  function defineLanguages(hljs) {
    // Recipe pseudocode: prose-like, so only the safe tokens are coloured
    // (comments, strings, numbers) — no keyword list that would catch English.
    hljs.registerLanguage("pseudo", function () {
      return {
        contains: [
          hljs.COMMENT("#", "$"),
          { className: "string", begin: '"', end: '"', illegal: "\\n" },
          { className: "string", begin: "'", end: "'", illegal: "\\n" },
          { className: "number", begin: "\\b0x[0-9a-fA-F]+\\b" },
          { className: "number", begin: "\\b\\d[\\d_]*(?:\\.\\d+)?\\b" },
        ],
      };
    });
    hljs.registerLanguage("mlir", function () {
      return {
        contains: [
          hljs.COMMENT("//", "$"),
          { className: "variable", begin: "%[\\w.$-]+" },
          { className: "symbol", begin: "[@^][\\w.$-]+" },
          { className: "type", begin: "\\b(?:tensor|memref|vector|index|none|i1|i4|i8|i16|i32|i64|ui8|f16|f32|bf16)\\b" },
          { className: "keyword", begin: "\\b(?:func|return|module|loc|attributes|dense|affine_map|unit)\\b" },
          { className: "string", begin: '"', end: '"', illegal: "\\n" },
          { className: "number", begin: "\\b0x[0-9a-fA-F]+\\b|\\b\\d[\\d_]*(?:\\.\\d+)?\\b" },
        ],
      };
    });
    hljs.registerLanguage("fbs", function () {
      return {
        keywords:
          "table struct enum union namespace root_type attribute include rpc_service file_identifier file_extension",
        contains: [
          hljs.COMMENT("//", "$"),
          { className: "type", begin: "\\b(?:bool|byte|ubyte|short|ushort|int|uint|long|ulong|float|double|string|int8|uint8|int16|uint16|int32|uint32|int64|uint64|float16|float32|float64)\\b" },
          { className: "string", begin: '"', end: '"', illegal: "\\n" },
          { className: "number", begin: "\\b0x[0-9a-fA-F]+\\b|\\b\\d+\\b" },
        ],
      };
    });
  }

  function relight() {
    if (typeof hljs === "undefined") return;
    defineLanguages(hljs);
    var map = {
      "language-text": "pseudo",
      "language-plaintext": "pseudo",
      "language-mlir": "mlir",
      "language-fbs": "fbs",
    };
    document.querySelectorAll("pre code").forEach(function (code) {
      for (var cls in map) {
        if (code.classList.contains(cls)) {
          try {
            code.innerHTML = hljs.highlight(map[cls], code.textContent, true).value;
            code.classList.add("hljs");
          } catch (e) { /* leave the block plain on any parser error */ }
          break;
        }
      }
    });
  }

  if (document.readyState === "complete") relight();
  else window.addEventListener("load", relight);
})();
