// Menu-bar and sidebar tweaks for the web edition.
//   1. Lock the guide to the dark "coal" theme (the picker is hidden in CSS).
//   2. "Back matter" and "Appendices" section headers in the sidebar. Those are
//      unnumbered suffix chapters (mdBook will not place a part title above
//      suffix chapters in SUMMARY), so the headers are inserted here, using
//      mdBook's own .part-title markup so they match the Part I-VIII headers.
//   3. A "download as PDF" button in the menu bar's right buttons, linking to the
//      rolling 'pdf-latest' GitHub release. The guide-build CI rebuilds that PDF
//      from the markdown on every push to main; the web/RTD edition does not build
//      the PDF itself.
//   4. A "Home" link in the menu bar's left buttons, back to the ANEForge landing
//      page (the guide is hosted standalone on Read the Docs).
(function () {
  function addSidebarHeaders() {
    var nav = document.querySelector(".sidebar .chapter, #sidebar ol.chapter");
    if (!nav) return;
    function headerBefore(hrefSuffix, title) {
      var existing = nav.querySelectorAll("li.part-title");
      for (var i = 0; i < existing.length; i++) {
        if (existing[i].textContent === title) return;     // already inserted
      }
      var a = nav.querySelector('a[href$="' + hrefSuffix + '"]');
      var li = a && a.closest("li");
      if (!li) return;
      var h = document.createElement("li");
      h.className = "part-title";
      h.textContent = title;
      li.parentNode.insertBefore(h, li);
    }
    headerBefore("back-matter/methodology.html", "Back matter");
    headerBefore("appendices/a-op-device-matrix.html", "Appendices");
    headerBefore("references.html", "References");
  }

  function lockTheme() {
    // The guide is locked to the dark "coal" theme (the picker is hidden in CSS).
    // Force it here so a stale saved theme from a previous visit cannot leave a
    // reader stuck on light with no way to switch back.
    var html = document.documentElement;
    ["light", "navy", "ayu", "rust"].forEach(function (c) { html.classList.remove(c); });
    if (!html.classList.contains("coal")) html.classList.add("coal");
    try { localStorage.setItem("mdbook-theme", "coal"); } catch (e) { /* ignore */ }
  }

  function addPdfButton() {
    var rb = document.querySelector("#menu-bar .right-buttons");
    if (!rb || document.querySelector("#pdf-button")) return;
    var a = document.createElement("a");
    a.id = "pdf-button";
    a.href = "https://github.com/sbryngelson/ane-guide/releases/download/pdf-latest/apple-neural-engine-guide.pdf";
    a.target = "_blank";
    a.rel = "noopener";
    a.title = "Download the whole guide as a PDF";
    a.setAttribute("aria-label", "Download as PDF");
    a.innerHTML = '<span class="pdf-button-label">Download PDF</span><i class="fa fa-file-pdf-o"></i>';
    rb.insertBefore(a, rb.firstChild);
  }

  function addHomeLink() {
    var bar = document.querySelector("#menu-bar .left-buttons");
    if (!bar || document.querySelector(".home-link")) return;
    var a = document.createElement("a");
    a.className = "home-link";
    a.href = "https://sbryngelson.github.io/ANEForge/";
    a.title = "ANEForge home";
    a.textContent = "Home";
    bar.appendChild(a);
  }

  function run() { lockTheme(); addHomeLink(); addSidebarHeaders(); addPdfButton(); }
  if (document.readyState !== "loading") run();
  else document.addEventListener("DOMContentLoaded", run);
})();
