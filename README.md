# Corpus

Official Turkish data protection texts used for retrieval experiments. About 22,000 words from 150 PDF pages.

| File | Document | Source |
|---|---|---|
| `kvkk_kanun_6698.txt` | 6698 sayılı Kişisel Verilerin Korunması Kanunu (consolidated, includes the 2024 amendments) | mevzuat.gov.tr |
| `yonetmelik_silme_yok_etme_anonim.txt` | Kişisel Verilerin Silinmesi, Yok Edilmesi veya Anonim Hale Getirilmesi Hakkında Yönetmelik | mevzuat.gov.tr |
| `teblig_aydinlatma.txt` | Aydınlatma Yükümlülüğünün Yerine Getirilmesinde Uyulacak Usul ve Esaslar Hakkında Tebliğ | mevzuat.gov.tr |
| `rehber_veri_guvenligi.txt` | Kişisel Veri Güvenliği Rehberi (Teknik ve İdari Tedbirler) | kvkk.gov.tr |
| `rehber_ozel_nitelikli.txt` | Özel Nitelikli Kişisel Verilerin İşlenmesine İlişkin Rehber | kvkk.gov.tr |

Word counts, page counts and source URLs are in `manifest.json`.

## How the text was cleaned

`scripts/extract_corpus.py` rebuilds everything in `text/` from the PDFs in `raw/`.

- Text is extracted with PyMuPDF, block by block.
- Page numbers, running headers, footnotes and footnote markers are removed using font size.
- Words hyphenated across line breaks are joined, and paragraphs split across pages are merged.
- Covers, tables of contents, the summary tables and bibliography of the security guide, and the amendment and staffing tables at the end of the legal texts are removed.
- Legal texts are restructured so every article starts on its own line with its title, and every clause `(1)` and item `a)` starts on a new line.
- Typographic ligatures are replaced with plain letters. Turkish characters and quotation marks are kept as they are.

## Known gaps

- Footnote markers are removed, so a few sentences in the guides read slightly differently from the PDF.
- Not included yet: Veri Sorumlusuna Başvuru Usul ve Esasları Hakkında Tebliğ, Aydınlatma Yükümlülüğünün Yerine Getirilmesi Rehberi.
