-- Render fenced ```mermaid code blocks to vector figures for the PDF build.
-- On GitHub the same blocks render natively; this filter only runs in build.sh.
-- mermaid-cli prints through Chrome to a cropped, vector PDF (text stays text and
-- the HTML node labels render), so figures are crisp at any scale. The figure is
-- shown at its natural width (capped to the text block) so the label font lands
-- near 10pt, matching the body. A caption is taken from a `%% caption:` comment
-- in the mermaid source (invisible on GitHub). If rendering fails the source
-- block is left in place so the build still completes.

local idx = 0
-- The text-block width (letter, 1in margins). Diagrams are included at their
-- natural 1:1 size so the mermaid font (13px ~ 9.75pt) lands at body size or
-- just under; a diagram wider than this is scaled down (its text gets smaller,
-- never larger), never scaled up.
local MAX_IN = 6.5

local function run(cmd)
  local ok = os.execute(cmd)
  return ok == true or ok == 0
end

local function pdf_width_in(pdf)
  local ok, out = pcall(pandoc.pipe, "pdfinfo", { pdf }, "")
  if not ok or not out then return nil end
  local w = out:match("Page size:%s+([%d%.]+)%s+x")
  return w and (tonumber(w) / 72) or nil
end

local function caption_of(text)
  return text:match("%%%%%s*[Cc]aption:%s*([^\n]+)")
end

function CodeBlock(el)
  if not el.classes:includes("mermaid") then return nil end
  idx = idx + 1
  run("mkdir -p build/diagrams")
  local base = "build/diagrams/d" .. idx
  local mmd, pdf = base .. ".mmd", base .. ".pdf"

  local fh = io.open(mmd, "w"); fh:write(el.text); fh:close()

  local render =
    "PUPPETEER_SKIP_DOWNLOAD=1 npx -y @mermaid-js/mermaid-cli@11"
      .. " -i " .. mmd .. " -o " .. pdf
      .. " -p pandoc/puppeteer.json -c pandoc/mermaid-config.json"
      .. " -f -b white >/dev/null 2>&1"

  if not run(render) then
    return el  -- toolchain unavailable: keep the source block
  end

  local win = pdf_width_in(pdf) or 4.0
  if win > MAX_IN then win = MAX_IN end

  -- The .tex is compiled from inside build/, so reference the image relative to it.
  local incpath = "diagrams/d" .. idx .. ".pdf"
  local latex = "\\begin{figure}[H]\\centering\n"
    .. "\\includegraphics[width=" .. string.format("%.3fin", win) .. "]{" .. incpath .. "}\n"
  local cap = caption_of(el.text)
  if cap then
    -- A trailing {#fig:slug} token in the caption becomes the cleveref label so
    -- the figure can be cross-referenced; on GitHub it is plain comment text.
    local label = cap:match("{#(fig:[%w:._%-]+)}")
    if label then cap = cap:gsub("%s*{#fig:[%w:._%-]+}%s*$", "") end
    latex = latex .. "\\caption{" .. cap .. "}\n"
    if label then latex = latex .. "\\label{" .. label .. "}\n" end
  end
  latex = latex .. "\\end{figure}"

  return pandoc.RawBlock("latex", latex)
end
