-- Style filter for the PDF build (no effect on GitHub rendering).
--
-- 1. Blockquote callouts -> tcolorbox. A blockquote whose first paragraph opens
--    with a bold run (for example **Recipe.**) becomes a titled callout box; any
--    other blockquote is a chapter's opening summary and becomes a left-ruled
--    summary box. Every blockquote in the guide is one of these two.
-- 2. Table header rows -> bold.
-- (Code-block highlighting for the in-house mlir and fbs tags is handled by the
-- custom syntax definitions pandoc/mlir.xml and pandoc/fbs.xml, passed to pandoc
-- with --syntax-definition; no language remap is needed here.)

local function latex_escape(s)
  return (s:gsub("([#%%&_%${}])", "\\%1"))
end

function BlockQuote(bq)
  local blocks = bq.content
  local label = nil
  local first = blocks[1]
  if first and first.t == "Para" and first.content[1] and first.content[1].t == "Strong" then
    label = pandoc.utils.stringify(first.content[1]):gsub("%s*%.%s*$", "")
    table.remove(first.content, 1)                       -- drop the bold label
    if first.content[1] and first.content[1].t == "Space" then
      table.remove(first.content, 1)                     -- and the space after it
    end
  end

  local open, close
  if label then
    open  = pandoc.RawBlock("latex", "\\begin{calloutbox}{" .. latex_escape(label) .. "}")
    close = pandoc.RawBlock("latex", "\\end{calloutbox}")
  else
    open  = pandoc.RawBlock("latex", "\\begin{summarybox}")
    close = pandoc.RawBlock("latex", "\\end{summarybox}")
  end

  local out = pandoc.Blocks({ open })
  for _, b in ipairs(blocks) do out:insert(b) end
  out:insert(close)
  return out
end

function Table(t)
  -- bold every header cell
  for _, row in ipairs(t.head.rows) do
    for _, cell in ipairs(row.cells) do
      cell.contents = cell.contents:map(function(blk)
        if blk.t == "Plain" or blk.t == "Para" then
          return pandoc.Plain({ pandoc.Strong(blk.content) })
        end
        return blk
      end)
    end
  end

  -- zebra-stripe alternate body rows by prepending \rowcolor to the first cell.
  -- A per-row \rowcolor (rather than xcolor's \rowcolors) survives longtable
  -- page breaks without the repeated header desyncing the stripe.
  for _, body in ipairs(t.bodies) do
    local n = 0
    for _, row in ipairs(body.body) do
      n = n + 1
      if n % 2 == 0 and row.cells[1] then
        local cell = row.cells[1]
        local blk = cell.contents[1]
        if blk and blk.content then
          table.insert(blk.content, 1, pandoc.RawInline("latex", "\\rowcolor{zebra}"))
        else
          table.insert(cell.contents, 1, pandoc.RawBlock("latex", "\\rowcolor{zebra}"))
        end
      end
    end
  end

  return t
end
