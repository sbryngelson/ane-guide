-- citations.lua — make the fixed inline citation keys clickable.
--
-- Inline citations are written as a bracketed key, for example [AppleANE] or
-- [Williams2009]. Pandoc keeps such a key inside a single Str token (any
-- trailing punctuation attaches to it), while a code token like `[Section]`
-- is a Code element and is left untouched. This filter rewrites each known
-- key into an internal link to its anchor in Appendix E.0, where the matching
-- span carries the id ref-<lowercased-key>.
--
-- Runs LAST in the chain (after the cross-reference filters) so the links it
-- creates are not re-interpreted as float or section references.

local keys = {
  AppleCoreML = true,
  AppleCoreMLTools = true,
  AppleANE = true,
  AppleAccelerate = true,
  AppleVision = true,
  AppleActiveDevices2026 = true,
  Williams2009 = true,
  Orion2026 = true,
  ANEForge2026 = true,
  -- E.1 external literature, cited in the Introduction's Related work
  Choi2013 = true,
  Ilic2014 = true,
  Ding2019 = true,
  Yang2020 = true,
  Verhelst2025 = true,
  Jouppi2017 = true,
  Ignatov2019 = true,
  Jayanth2024 = true,
  Prashanthi2025 = true,
  Fanariotis2025 = true,
  Tummalapalli2026 = true,
  Bi2026 = true,
  Xu2025 = true,
  Moon2025 = true,
  Chen2025 = true,
  Benazir2026 = true,
  Hubner2025 = true,
  tinygrad = true,
  eilnANE = true,
  Singh2026 = true,
  libane = true,
  whispercpp = true,
  Hollemans = true,
  AppleANETransformers = true,
  Plyenkov2019 = true,
  CommunityANE = true,
  Zeus2025 = true,
}

function Str(el)
  local text = el.text
  if not text:find("%[%w+%]") then
    return nil
  end
  local out = {}
  local emit_from = 1  -- start of text not yet emitted
  local search_from = 1 -- where the next bracket search begins
  local changed = false
  while true do
    local s, e, key = text:find("%[(%w+)%]", search_from)
    if not s then
      break
    end
    if keys[key] then
      changed = true
      if s > emit_from then
        table.insert(out, pandoc.Str(text:sub(emit_from, s - 1)))
      end
      table.insert(out, pandoc.Link(pandoc.Str("[" .. key .. "]"), "#ref-" .. key:lower()))
      emit_from = e + 1
    end
    search_from = e + 1
  end
  if not changed then
    return nil
  end
  if emit_from <= #text then
    table.insert(out, pandoc.Str(text:sub(emit_from)))
  end
  return out
end
