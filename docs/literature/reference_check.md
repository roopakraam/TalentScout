# Reference check (T8.4)

Every reference in `report/chapters/11_references.md` and `paper/references.bib` was checked on 14 Nov 2026.

* **20 references with a DOI** were resolved through the Crossref API; authors, year, volume, issue and pages
  match the publisher record for all 20.
* **Corrected:** the author names of Tang, Wei and Tan (2026) are Qingrong Tang, Xiufang Wei and Bo Tan
  (the BibTeX entry had wrong first names; the initials in the report were right). The issue number (1)
  of that article is inferred from the journal's volume layout; Crossref does not list it.
* **Without a DOI** (conference proceedings of NeurIPS, JMLR articles, books): Bergstra and Bengio (2012),
  Bergstra et al. (2011), Cawley and Talbot (2010), Grinsztajn et al. (2022), Holm (1979), Ke et al. (2017),
  Lundberg and Lee (2017), Holland (1975), Luke (2013), Sivanandam and Deepa (2018). These are standard
  references; volume and page numbers follow the publishers' proceedings pages. The automated Semantic Scholar
  lookup was rate-limited, so a manual check on Google Scholar is still worth doing before submission.
* Emary, Zawbaa and Hassanien (2016) is paywalled; the bGWO2 transfer function and the α = 0.99 weighting
  follow how the method is described in later papers (see `paper_notes.md`). Check them against the full
  text if you have library access.
