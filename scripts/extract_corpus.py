"""Build corpus/text and corpus/manifest.json from the PDFs in corpus/raw.

Run from the repo root: python scripts/extract_corpus.py
"""
import pymupdf, re, collections, json, unicodedata

DOCS = [
 dict(id="kvkk_kanun_6698", file="kvkk_kanun_6698.pdf",
      title="6698 sayılı Kişisel Verilerin Korunması Kanunu",
      source="https://mevzuat.gov.tr/MevzuatMetin/1.5.6698.pdf", kind="kanun",
      skip_pages=[21], cut=[("(I) SAYILI CETVEL", None)],
      min_ratio=0.95),  # amendment footnotes are 11pt, body is 12pt
 dict(id="yonetmelik_silme_yok_etme_anonim", file="yonetmelik_silme_yok_etme_anonim.pdf",
      title="Kişisel Verilerin Silinmesi, Yok Edilmesi veya Anonim Hale Getirilmesi Hakkında Yönetmelik",
      source="mevzuat.gov.tr (GeneratePdf)", kind="yonetmelik", skip_pages=[],
      cut=[("Yönetmeliğin Yayımlandığı Resmî Gazete", None)]),
 dict(id="teblig_aydinlatma", file="teblig_aydinlatma.pdf",
      title="Aydınlatma Yükümlülüğünün Yerine Getirilmesinde Uyulacak Usul ve Esaslar Hakkında Tebliğ",
      source="mevzuat.gov.tr (GeneratePdf)", kind="teblig", skip_pages=[],
      cut=[("Tebliğin Yayımlandığı Resmî Gazete", None)]),
 dict(id="rehber_veri_guvenligi", file="rehber_veri_guvenligi.pdf",
      title="Kişisel Veri Güvenliği Rehberi (Teknik ve İdari Tedbirler)",
      source="https://www.kvkk.gov.tr/yayinlar/veri_guvenligi_rehberi.pdf", kind="rehber", skip_pages=None,
      cut=[("", "1.1. Amaç ve Dayanak"), ("4.1. Teknik Tedbirler Özet Tablosu", None)]),
 dict(id="rehber_ozel_nitelikli", file="rehber_ozel_nitelikli.pdf",
      title="Özel Nitelikli Kişisel Verilerin İşlenmesine İlişkin Rehber",
      source="https://www.kvkk.gov.tr/SharedFolderServer/CMSFiles/70f95c73-06a2-44dc-81e9-34201bdd7f5c.pdf", kind="rehber", skip_pages=None, min_ratio=0.95,
      cut=[("", "I. GİRİŞ")]),
]

TOC_RE = re.compile(r"İÇİNDEKİLER")
PAGENUM_RE = re.compile(r"^\s*([0-9]{1,3}|[ivxlcdm]{1,6})\s*$", re.I)
RUNHEAD_RE = re.compile(r"^\s*\d{1,3}\s*\|\s*Özel Nitelikli Kişisel Verilerin İşlenmesine İlişkin Rehber\s*$")

def body_size(doc):
    c = collections.Counter()
    for p in doc:
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    c[round(s["size"], 1)] += len(s["text"].strip())
    return c.most_common(1)[0][0]

def join_lines(lines):
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif re.search(r"[a-zçğıöşü]-$", out) and re.match(r"^[a-zçğıöşü]", ln):
            out = out[:-1] + ln
        else:
            out = out + " " + ln
    return re.sub(r"[ \t  ]+", " ", out).strip()

def extract(d):
    doc = pymupdf.open("corpus/raw/" + d["file"])
    bs = body_size(doc)
    lo = d.get("min_ratio", 0.8)
    items = []  # (text, is_heading)
    for i, p in enumerate(doc, start=1):
        if d["skip_pages"] and i in d["skip_pages"]:
            continue
        txt = p.get_text()
        if TOC_RE.search(txt) or txt.count("....") > 5:
            continue
        for b in p.get_text("dict")["blocks"]:
            lines, sizes, chars = [], 0.0, 0
            for l in b.get("lines", []):
                spans = [s for s in l["spans"] if bs * lo <= s["size"] <= bs * 3]
                t = "".join(s["text"] for s in spans)
                for s in spans:
                    n = len(s["text"].strip()); sizes += s["size"] * n; chars += n
                if t.strip():
                    lines.append(t)
            para = join_lines(lines)
            if not para or PAGENUM_RE.match(para) or RUNHEAD_RE.match(para) or re.search(r"\|\s*\d{1,3}$", para):
                continue
            heading = chars > 0 and (sizes / chars) > bs * 1.05
            items.append([para, heading])
    merged = []
    for para, h in items:
        if merged and merged[-1][1] and re.match(r"^[a-zçğıöşü]", para):
            merged[-1][0] = join_lines([merged[-1][0], para]); continue
        if merged and not h and not merged[-1][1]:
            prev = merged[-1][0]
            if re.match(r"^[a-zçğıöşü]", para) or not re.search(r"[.:;!?\u201d\"]$", prev):
                if re.search(r"[a-zçğıöşü]-$", prev) and re.match(r"^[a-zçğıöşü]", para):
                    merged[-1][0] = prev[:-1] + para
                else:
                    merged[-1][0] = join_lines([prev, para])
                continue
        merged.append([para, h])
    paras = [p for p, _ in merged]
    text = "\n\n".join(paras)
    text = unicodedata.normalize("NFC", text)
    for a, b in {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": "", "\u200b": ""}.items():
        text = text.replace(a, b)
    text = text.replace("\n\nürkiye Cumhuriyeti Anayasası", "\n\nTürkiye Cumhuriyeti Anayasası")
    for start, end in d.get("cut", []):
        a = text.find(start)
        b = text.find(end, a) if end else len(text)
        if a >= 0 and b >= 0:
            text = text[:a] + text[b:]
    if d["kind"] in ("kanun", "yonetmelik", "teblig"):
        text = legal_structure(text)
    return text.strip(), doc.page_count, bs

ORD = "BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU"
def legal_structure(t):
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"\s+((?:%s) BÖLÜM)\s+" % ORD, r"\n\n\1\n", t)
    t = re.sub(r"(?<=[.:;\n])\s*([A-ZÇĞİÖŞÜ][^.\n]{2,160}?)\s+((?:GEÇİCİ |EK )?MADDE \d+)\s*[-\u2013]\s*", r"\n\n\1\n\2 - ", t)
    t = re.sub(r"(?<=[.:;)])\s+\((\d+)\)\s", r"\n(\1) ", t)
    t = re.sub(r"(?<=[,;:.])\s+([a-zçğıöşü]{1,2}\))\s", r"\n\1 ", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

manifest = []
for d in DOCS:
    text, n, bs = extract(d)
    open(f"corpus/text/{d['id']}.txt", "w", encoding="utf-8").write(text + "\n")
    manifest.append(dict(id=d["id"], title=d["title"], kind=d["kind"], source=d["source"],
                         downloaded="2026-10-01", pages=n, words=len(text.split()), chars=len(text)))
    print(d["id"], n, "pages", len(text.split()), "words", "body", bs)
json.dump(manifest, open("corpus/manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("TOTAL words", sum(m["words"] for m in manifest), "pages", sum(m["pages"] for m in manifest))
