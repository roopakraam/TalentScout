# Paper draft

**Word version (main):** `TalentScout_v2_Paper.docx` and `.pdf` (7 pages), built from `chapters/paper.md` with `python report/build_report.py --paper`.
The LaTeX version below is kept for venues that require it.

`main.tex` is a condensed version of the report in Springer LNCS format (about 2,500 words,
3 figures, 2 tables, 18 references). The figures in `figures/` are the vector PDFs from
`results/figures/`, and `references.bib` was checked against Crossref.

## Compiling

LaTeX is not installed on the project machine, so the draft has been checked statically
(citations, labels, environments, figure files) but not compiled. On Overleaf: create a new
project, upload `main.tex`, `references.bib` and the `figures/` folder, and compile with pdfLaTeX.
Locally: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. The `llncs` class and
the `splncs04` bibliography style come with TeX Live and Overleaf; if they are missing, download
the Springer LNCS author kit.

## Where to submit

The plan (§8) lists these venues. Checked on 14 Nov 2026; confirm dates on the venue sites before submitting.

| Venue | Format and limit | Timing |
| --- | --- | --- |
| MLSA workshop (Machine Learning and Data Mining for Sports Analytics) at ECML PKDD | LNCS; this draft is already in that format | The 2026 edition took papers through the ECML PKDD workshop track, with the track deadline in early June ([call](https://groups.google.com/g/ml-news/c/YOVJDCgGYTk), [workshop track](https://ecmlpkdd.org/2026/submissions-workshop-track/)); expect the 2027 call in spring 2027 |
| GECCO 2027 (Genetic and Evolutionary Computation Conference) | ACM format; full papers 8 pages excluding references, poster-only papers 4 pages ([call](https://gecco-2027.sigevo.org/Call-for-Papers)) | Main deadlines not yet confirmed; the 2026 student workshop deadline was 3 April 2026 ([2026 student workshop](https://gecco-2026.sigevo.org/Call+for+Student+Workshop)) |
| Journal of Sports Analytics; International Journal of Computer Science in Sport | journal formats | rolling submission |

MLSA is the closest fit for the sports angle and needs no reformatting. For GECCO the text would
have to move to the ACM template and be cut to 8 pages.

Before submitting: add your e-mail address under `\institute`, and anonymise the draft if the
venue reviews double-blind.
