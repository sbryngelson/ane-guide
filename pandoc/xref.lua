-- Turn in-text "chapter N" and "Appendix X" references into internal hyperlinks.
-- Chapters and appendices are the numbered level-1 headers; numbering.lua has
-- already stripped the manual prefix and tagged the appendix headers, so the maps
-- are built by counting numbered headers in document order (skipping the
-- unnumbered front/back matter), which matches LaTeX's own chapter numbering.

local chap = {}   -- ["9"]  = identifier of chapter 9
local app = {}    -- ["A"]  = identifier of Appendix A

local function collect(blocks)
  local cn, an = 0, 0
  pandoc.walk_block(pandoc.Div(blocks), {
    Header = function(h)
      if h.level == 1 and h.identifier ~= "" then
        if h.classes:includes("unnumbered") then return end
        if h.classes:includes("appendix") then
          an = an + 1
          app[string.char(string.byte("A") + an - 1)] = h.identifier
        else
          cn = cn + 1
          chap[tostring(cn)] = h.identifier
        end
      end
    end,
  })
end

local function linkify(inlines)
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #inlines do
    local w, sp, num = inlines[i], inlines[i + 1], inlines[i + 2]
    local handled = false
    if w and w.t == "Str" and sp and sp.t == "Space" and num and num.t == "Str" then
      local word = w.text:lower()
      if (word == "chapter" or word == "chapters") then
        local key = num.text:match("^(%d+)")
        if key and chap[key] then
          out:insert(w); out:insert(sp)
          out:insert(pandoc.Link(pandoc.Inlines({ pandoc.Str(key) }), "#" .. chap[key]))
          local rest = num.text:sub(#key + 1)
          if rest ~= "" then out:insert(pandoc.Str(rest)) end
          i = i + 3; handled = true
        end
      elseif word == "appendix" then
        local key = num.text:match("^([A-Z])")
        if key and app[key] then
          out:insert(w); out:insert(sp)
          out:insert(pandoc.Link(pandoc.Inlines({ pandoc.Str(key) }), "#" .. app[key]))
          local rest = num.text:sub(2)
          if rest ~= "" then out:insert(pandoc.Str(rest)) end
          i = i + 3; handled = true
        end
      end
    end
    if not handled then out:insert(w); i = i + 1 end
  end
  return out
end

function Pandoc(doc)
  collect(doc.blocks)
  return doc:walk({ Inlines = linkify })
end
