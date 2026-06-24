-- PDF numbering setup. Runs only in build.sh; the source headings keep their
-- manual "N." for GitHub, which renders no automatic numbering of its own.
--
-- 1. Mark the structural front-matter, back-matter, and handoff level-1
--    headings as unnumbered, so LaTeX's chapter counter advances only over the
--    real chapters (1..36) and then the appendices (A..F).
-- 2. Strip the manual "N." / "Appendix X." prefix from a numbered chapter title
--    so the auto chapter number that LaTeX now prints does not double it; tag the
--    appendix headings so xref.lua can letter them.

local UNNUMBERED = {
  "^Interlude",
  "^References$",
  "^Methodology$",
  "^Open questions$",
  "^Statements$",
  "^Revision history$",
}

-- Headers are visited in document order, so a flag carries "inside an unnumbered
-- chapter" down to the child sections, which must be unnumbered too (otherwise a
-- section in the front matter would number against chapter 0).
local in_unnumbered = false

local function mark(h)
  if not h.classes:includes("unnumbered") then h.classes:insert("unnumbered") end
end

function Header(h)
  if h.level ~= 1 then
    if in_unnumbered then mark(h) end
    return h
  end

  local text = pandoc.utils.stringify(h)

  for _, pat in ipairs(UNNUMBERED) do
    if text:match(pat) then
      mark(h)
      in_unnumbered = true
      return h
    end
  end

  in_unnumbered = false

  local appendix = text:match("^Appendix%s+%a+%.%s+(.*)$")
  if appendix then
    h.classes:insert("appendix")
    h.content = pandoc.Inlines({ pandoc.Str(appendix) })
    return h
  end

  local chapter = text:match("^%d+%.%s+(.*)$")
  if chapter then
    h.content = pandoc.Inlines({ pandoc.Str(chapter) })
    return h
  end

  return nil
end
