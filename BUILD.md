# Building the guide

`build.sh` renders the Markdown sources into `build/ane-guide.pdf` through pandoc
and XeLaTeX, and `check.sh` plus markdownlint validate them. The list below is
everything a fresh machine needs.

## Dependencies

| Tool | Used for |
| --- | --- |
| pandoc | Markdown to LaTeX conversion |
| XeLaTeX (MacTeX or TeX Live) | typesetting the PDF |
| Node and npx | mermaid-cli (diagrams) and markdownlint-cli2, fetched on demand |
| Google Chrome | mermaid-cli renders diagrams through it (see `pandoc/puppeteer.json`) |
| poppler | `pdfinfo` and `pdftotext`, used by the diagram filter and the checks |

The LaTeX preamble (`pandoc/header.tex`) uses these packages, all present in a full
MacTeX or TeX Live install: `tcolorbox`, `fvextra`, `framed`, `sectsty`, `fancyhdr`,
`microtype`, `colortbl`, `newfloat`, `caption`, `etoolbox`, `xcolor`, and `hyperref`.

## Fonts

The build uses three font families, needed by both XeLaTeX and the mermaid diagram
renderer (diagrams are set in Source Sans Pro so their labels match the document):

- Source Sans Pro: headings and diagram labels
- Source Code Pro: code
- Charter: body text, which is included with macOS

On macOS, Source Sans Pro and Source Code Pro install as Homebrew casks (see
Setup and build below); Charter ships with macOS.

## Setup and build

On macOS with Homebrew, install the toolchain and fonts once:

```sh
brew install pandoc poppler node
brew install --cask mactex-no-gui google-chrome
brew install --cask font-source-sans-pro font-source-code-pro
```

Then build:

```sh
bash guide/build.sh         # writes guide/build/ane-guide.pdf
```

With BasicTeX in place of MacTeX, add the LaTeX packages once:

```sh
sudo tlmgr update --self
sudo tlmgr install tcolorbox fvextra framed sectsty fancyhdr microtype \
  colortbl newfloat caption etoolbox xcolor
```

If Chrome is installed somewhere other than `/Applications/Google Chrome.app`, edit
the `executablePath` in `pandoc/puppeteer.json`.

## Checks

```sh
bash guide/check.sh                        # editorial invariants
npx -y markdownlint-cli2 'guide/**/*.md'   # Markdown formatting
```

Both run automatically on commit through the pre-commit hook in `.githooks/`,
activated once per clone with `git config core.hooksPath .githooks`.
