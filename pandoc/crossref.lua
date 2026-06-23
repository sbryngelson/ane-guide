-- Float cross-references for the PDF build (cleveref). Runs only in build.sh; on
-- GitHub the reference is a plain Markdown link and the caption keeps its label
-- token invisible at the end of the line.
--
-- Authoring:
--   * Label a float by ending its caption with a token:
--       Table: The roofline constants. {#tbl:roofline}
--       Listing: ... {#lst:dispatch}      (handled in listing.lua)
--       %% caption: ... {#fig:soc}         (handled in mermaid.lua)
--   * Reference it with a Markdown link whose target is the label:
--       the [roofline table](#tbl:roofline) shows ...   -> "table 9"   (\cref)
--       [Table](#tbl:roofline) lists ...                -> "Table 9"   (\Cref)
--     A lowercase first letter in the link text yields \cref (mid-sentence),
--     an uppercase one yields \Cref (sentence start). The link text itself is
--     only the GitHub fallback; in the PDF cleveref supplies the name and number.

-- Pull a trailing {#tbl:slug} token out of a caption's inline list, returning the
-- label and leaving the visible caption text clean.
local function strip_tbl_label(inlines)
  for i = #inlines, 1, -1 do
    local el = inlines[i]
    if el.t == "Str" then
      local label = el.text:match("^{#(tbl:[%w:._%-]+)}$")
      if label then
        table.remove(inlines, i)
        if inlines[i - 1] and inlines[i - 1].t == "Space" then
          table.remove(inlines, i - 1)
        end
        return label
      end
    end
  end
  return nil
end

local function table_label(t)
  local blocks = t.caption and t.caption.long
  if not blocks then return end
  for _, blk in ipairs(blocks) do
    if blk.content then
      local label = strip_tbl_label(blk.content)
      if label then
        blk.content:insert(pandoc.RawInline("latex", "\\label{" .. label .. "}"))
        return
      end
    end
  end
end

-- A reference is a link to a float label (#tbl:/#fig:/#lst:) or to a section,
-- chapter, or appendix anchor (a heading identifier). Either way cleveref supplies
-- the name and the number; the link text's first letter picks \cref vs \Cref.
local function make_ref(l, headers)
  local tgt = l.target
  if tgt:sub(1, 1) ~= "#" then return nil end
  local key = tgt:sub(2)
  local is_float = key:match("^tbl:") or key:match("^fig:") or key:match("^lst:")
  if not (is_float or headers[key]) then return nil end
  local first = pandoc.utils.stringify(l.content):match("%a")
  local cmd = (first and first:match("%u")) and "\\Cref" or "\\cref"
  return pandoc.RawInline("latex", cmd .. "{" .. key .. "}")
end

function Pandoc(doc)
  local headers = {}
  doc:walk({ Header = function(h) if h.identifier ~= "" then headers[h.identifier] = true end end })
  return doc:walk({
    Table = function(t) table_label(t); return t end,
    Link = function(l) return make_ref(l, headers) end,
  })
end
