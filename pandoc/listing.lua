-- Turn a code block followed by a `Listing: <caption>` paragraph into a numbered
-- listing float ("Listing N. <caption>") with the caption above the code.
-- On GitHub the `Listing:` line shows as plain text; this filter only runs for PDF.

local function is_caption(b)
  return b and b.t == "Para" and #b.content > 0
     and b.content[1].t == "Str" and b.content[1].text == "Listing:"
end

-- The caption is stringified to plain text and injected into a raw \caption{},
-- so LaTeX specials (notably the underscore in identifiers like kANE_UKNOWN)
-- must be escaped here; pandoc does not escape inside a RawBlock.
local function latex_escape(s)
  s = s:gsub("\\", "\\textbackslash{}")
  s = s:gsub("([&%%$#_{}])", "\\%1")
  s = s:gsub("~", "\\textasciitilde{}")
  s = s:gsub("%^", "\\textasciicircum{}")
  return s
end

-- Returns the escaped caption text and an optional cleveref label. A trailing
-- {#lst:slug} token in the caption is pulled out as the label (\label below) so
-- the listing can be cross-referenced; on GitHub it is plain caption text.
local function caption_text(b)
  local rest = {}
  for i = 3, #b.content do rest[#rest + 1] = b.content[i] end  -- drop "Listing:" + Space
  local s = pandoc.utils.stringify(pandoc.Para(rest))
  local label = s:match("{#(lst:[%w:._%-]+)}")
  if label then s = s:gsub("%s*{#lst:[%w:._%-]+}%s*$", "") end
  return latex_escape(s), label
end

function Blocks(blocks)
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    if b.t == "CodeBlock" and is_caption(blocks[i + 1]) then
      local cap, label = caption_text(blocks[i + 1])
      local labeltex = label and ("\\label{" .. label .. "}") or ""
      -- Count lines: a float ([H]) cannot break across pages, so a long block
      -- overflows the page bottom. Long listings are set as a non-float captioned
      -- block (caption via \captionof keeps the same "Listing N" counter and the
      -- cleveref label) whose code body breaks across pages; short ones stay a
      -- pinned float so caption and code travel together.
      local nlines = 1
      for _ in b.text:gmatch("\n") do nlines = nlines + 1 end
      if nlines > 38 then
        out:insert(pandoc.RawBlock("latex",
          "\\par\\addvspace{\\medskipamount}\\noindent\\captionof{listing}{" .. cap .. "}"
          .. labeltex .. "\\nopagebreak\\par\\vspace{1pt}"))
        out:insert(b)
        out:insert(pandoc.RawBlock("latex", "\\addvspace{\\medskipamount}"))
      else
        out:insert(pandoc.RawBlock("latex", "\\begin{listing}[H]\n\\caption{" .. cap .. "}" .. labeltex))
        out:insert(b)
        out:insert(pandoc.RawBlock("latex", "\\end{listing}"))
      end
      i = i + 2
    else
      out:insert(b)
      i = i + 1
    end
  end
  return out
end
