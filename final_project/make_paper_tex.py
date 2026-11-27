"""Render the paper in LaTeX from the same content as make_paper.py.

make_paper.py builds a ReportLab story. This script runs it with the PDF writer replaced by a recorder,
converts each heading, paragraph, table and figure to LaTeX, writes deliverables/CS4260_Final_Paper.tex
and compiles it with latexmk (pdflatex). make_paper.py is not modified, so its text and every computed
number carry over unchanged. Two presentation changes only: the contribution statement moves to the end
next to the AI-use acknowledgment, and Section 7's model is also written as equations.

    python final_project/make_paper_tex.py
"""

import html
import re
import shutil
import subprocess
from pathlib import Path

from reportlab.platypus import Image, KeepTogether, Paragraph, Table

import make_paper

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables"
NAME = "CS4260_Final_Paper_latex"

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype,graphicx,booktabs,array,amsmath,xcolor,newunicodechar,fancyhdr,titlesec}
\usepackage[font=small,labelformat=empty,skip=4pt]{caption}
\definecolor{accent}{HTML}{126078}
\usepackage[colorlinks,urlcolor=accent,linkcolor=accent]{hyperref}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\titleformat{\section}{\normalfont\large\bfseries\color{accent}}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{10pt}{4pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\small\color{gray}CS4260 · Fall 2026}
\fancyfoot[R]{\small\color{gray}\thepage}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[L]{\small\color{gray}CS4260 · Fall 2026}\fancyfoot[R]{\small\color{gray}\thepage}}
\begin{document}
"""

# Section 7's model in equation form, placed after the paragraph that states it in words.
EQUATIONS = r"""
\begin{gather*}
R = \max(0, R^{*}), \qquad R^{*} \sim \mathcal{N}\!\left(\mu, \sigma^{2}\right), \qquad \mu = w_0 + \mathbf{w}^{\top}\mathbf{x},\\
\log \sigma = \gamma_0 + \gamma_1 \sin\theta + \gamma_2 \cos\theta, \qquad \theta = 2\pi\,\frac{\text{day of year} - 1}{365.25},\\[4pt]
\min_{w_0,\,\mathbf{w},\,\boldsymbol{\gamma}}\;
-\sum_{i:\,R_i>0} \log\!\left[\frac{1}{\sigma_i}\,\phi\!\left(\frac{R_i-\mu_i}{\sigma_i}\right)\right]
-\sum_{i:\,R_i=0} \log \Phi\!\left(\frac{-\mu_i}{\sigma_i}\right)
+ \lambda \lVert \mathbf{w} \rVert_2^{2},\\[4pt]
P(a \le Y < b) = F(b - M) - F(a - M), \qquad
F(r) = \begin{cases} \Phi\!\left(\frac{r-\mu}{\sigma}\right) & r > 0,\\ 0 & r \le 0. \end{cases}
\end{gather*}
"""
EXTRA_AFTER = {"The weights maximize the likelihood": EQUATIONS}

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


class Recorder:
    """Stands in for SimpleDocTemplate: keeps the story instead of writing a PDF."""
    story = None

    def __init__(self, *args, **kwargs):
        pass

    def build(self, content, **kwargs):
        Recorder.story = content


def escape(text):
    return "".join(SPECIAL.get(ch, ch) for ch in text)


def latex(markup):
    """ReportLab paragraph markup (<b>, <i>, <br/>, <link href>) to LaTeX."""
    out, position = [], 0
    for match in re.finditer(r"<(/?)(br\s*/?|b|i|link)((?:\s[^>]*)?)>", markup):
        out.append(escape(html.unescape(markup[position:match.start()])))
        closing, tag, attributes = match.groups()
        if tag in ("b", "i"):
            out.append("}" if closing else (r"\textbf{" if tag == "b" else r"\emph{"))
        elif tag.startswith("br"):
            out.append(r"\\ ")
        elif closing:
            out.append("}")
        else:
            url = re.search(r"href=['\"]([^'\"]+)", attributes).group(1)
            out.append(r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{")
        position = match.end()
    out.append(escape(html.unescape(markup[position:])))
    return "".join(out)


def flatten(items):
    for item in items:
        if isinstance(item, KeepTogether):
            yield from flatten(item._content)
        else:
            yield item


def table(flowable, caption):
    rows = flowable._cellvalues
    widths = flowable._argW
    total = sum(widths)
    spec = []
    for column, width in enumerate(widths):
        style = rows[-1][column].style.name
        numeric = all(re.fullmatch(r"[+\-−]?\d+(\.\d+)?%?", html.unescape(row[column].text).strip())
                      for row in rows[1:])
        align = r"\raggedleft" if ("Right" in style or style == "Number" or numeric) else r"\raggedright"
        spec.append(rf">{{{align}\arraybackslash}}p{{\dimexpr {width / total:.4f}\linewidth-2\tabcolsep\relax}}")
    lines = [r"\begin{table}[!htb]", r"\centering\small", r"\begin{tabular}{@{}" + "".join(spec) + "@{}}", r"\toprule"]
    lines.append(" & ".join(r"\textbf{" + latex(cell.text) + "}" for cell in rows[0]) + r" \\")
    lines.append(r"\midrule")
    lines += [" & ".join(latex(cell.text) for cell in row) + r" \\" for row in rows[1:]]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\caption{" + latex(caption) + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def figure(flowable, caption):
    lines = [r"\begin{figure}[!htb]", r"\centering",
             r"\includegraphics[width=\linewidth]{" + Path(flowable.filename).resolve().as_posix() + "}"]
    if caption:
        lines.append(r"\caption{" + latex(caption) + "}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render(story):
    items = [item for item in flatten(story) if isinstance(item, (Paragraph, Table, Image))]
    body, contribution, used = [], [], set()
    title = authors = None
    section = 0
    target = body
    for index, item in enumerate(items):
        if index in used:
            continue
        if isinstance(item, (Table, Image)):
            caption = None
            following = items[index + 1] if index + 1 < len(items) else None
            if isinstance(following, Paragraph) and re.match(r"<b>(Table|Figure) \d+\.</b>", following.text):
                caption = following.text
                used.add(index + 1)
            target.append(table(item, caption) if isinstance(item, Table) else figure(item, caption))
            continue
        style, text = item.style.name, item.text
        if style == "Title":
            title = latex(text)
        elif authors is None and title is not None and style == "Body":
            authors = latex(text)
        elif style == "Heading":
            plain = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
            target = body
            numbered = re.match(r"(\d+)\.\s+(.*)", plain)
            if plain == "Abstract":
                summary = items[index + 1]
                used.add(index + 1)
                body.append(r"\begin{abstract}" + "\n" + latex(summary.text) + "\n" + r"\end{abstract}")
            elif plain == "Contribution statement":
                target = contribution
                contribution.append(r"\section*{Contribution statement}")
            elif numbered:
                section += 1
                if int(numbered.group(1)) != section:
                    raise ValueError(f"section numbering out of order at {plain}")
                body.append(r"\section{" + escape(numbered.group(2)) + "}")
            elif plain == "References":
                body.extend(contribution)
                contribution = []
                body.append(r"\section*{References}" + "\n" + r"\small")
            else:
                body.append(r"\section*{" + escape(plain) + "}")
        else:
            target.append(latex(text))
            for start, extra in EXTRA_AFTER.items():
                if text.startswith(start):
                    target.append(extra)
    head = [r"\title{\vspace{-2em}\textbf{" + title + "}}", r"\author{" + authors + "}", r"\date{}",
            r"\maketitle", r"\thispagestyle{fancy}"]
    return PREAMBLE + "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n\\end{document}\n"


def main():
    original = make_paper.SimpleDocTemplate
    make_paper.SimpleDocTemplate = Recorder
    try:
        make_paper.main()
    finally:
        make_paper.SimpleDocTemplate = original
    OUT.mkdir(exist_ok=True)
    tex = OUT / f"{NAME}.tex"
    tex.write_text(render(Recorder.story), encoding="utf-8")
    if shutil.which("latexmk"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={OUT}", str(tex)],
                       check=True, stdout=subprocess.DEVNULL)
        print(OUT / f"{NAME}.pdf")
    else:
        print(f"Wrote {tex}; compile it with pdflatex (run twice).")


if __name__ == "__main__":
    main()
