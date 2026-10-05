"""Answer a question from the KVKK corpus: HybridRetriever top-5 chunks -> LLM -> Turkish answer with citations.

Run from the repo root: python scripts/answer.py "soru" [--dry-run]
--dry-run prints the prompt and makes no LLM call.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 5
NO_INFO = "Bu konuda verilen metinlerde bilgi yok."

SYSTEM = f"""Sen KVKK mevzuatıyla ilgili soruları yanıtlayan bir asistansın. Sana numaralı metin parçaları ve bir soru verilecek.

Kurallar:
1. Yalnızca verilen metinlerdeki bilgilerle cevap ver. Kendi bilgini, başka kaynakları veya tahmini ekleme.
2. Her iddianın sonuna kaynağını köşeli parantezle yaz: kanun, yönetmelik ve tebliğ için [belge, Madde n], rehberler için [belge, bölüm numarası], örneğin [kvkk_kanun_6698, Madde 12] veya [rehber_veri_guvenligi, 3.1]. Belge adını ve madde/bölüm etiketini metin başlığında verildiği gibi yaz. Her köşeli parantezde tek bir kaynak olsun.
3. Bir metin parçası birden fazla maddeyi kapsıyorsa, bilginin geçtiği maddeyi metindeki "MADDE n" başlıklarına bakarak seç.
4. Sorunun cevabı verilen metinlerde yoksa yalnızca şunu yaz: "{NO_INFO}"
5. Kısa ve sade yaz. Metinde ne yazdığını aktar; hukuki tavsiye verme, yorum veya öneri ekleme."""

CITATION_RE = re.compile(r"\[([a-z0-9_]+)\s*,\s*([^\]\[]+?)\s*\]")


def chunk_header(i: int, c: dict) -> str:
    start = c["section"] or "başlık öncesi"
    covered = ", ".join(c["sections"]) or "-"
    return f"[Metin {i}] belge: {c['doc']} | başladığı yer: {start} | kapsadığı madde/bölümler: {covered}"


def build_prompt(question: str, chunks: list[dict]) -> str:
    parts = [f"{chunk_header(i, c)}\n{c['text']}" for i, c in enumerate(chunks, 1)]
    return "Metinler:\n\n" + "\n\n".join(parts) + f"\n\nSoru: {question}"


def _label_key(s: str) -> str:
    return " ".join(s.split()).casefold()


def parse_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Unique [doc, label] citations, each with the ranks of retrieved chunks that cover that label."""
    out = {}
    for doc, label in CITATION_RE.findall(answer):
        key = (doc, _label_key(label))
        if key in out:
            continue
        ranks = [c["rank"] for c in chunks
                 if c["doc"] == doc and key[1] in {_label_key(s) for s in c["sections"]}]
        out[key] = {"doc": doc, "section": label, "chunk_ranks": ranks}
    return list(out.values())


def is_no_info(answer: str) -> bool:
    return NO_INFO.rstrip(".").casefold() in answer.casefold()


def load_retriever():
    from retriever import HybridRetriever
    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
    return HybridRetriever(texts)


def answer(question: str, retriever, k=TOP_K) -> dict:
    from llm import MODEL, generate
    chunks = retriever.search(question, k=k)
    out = generate(build_prompt(question, chunks), SYSTEM)
    return {"question": question, "model": MODEL, "retrieved": chunks, "answer": out["text"],
            "citations": parse_citations(out["text"], chunks), "no_info": is_no_info(out["text"]),
            "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"], "cost_usd": out["cost_usd"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt, do not call the LLM")
    args = parser.parse_args()

    retriever = load_retriever()
    if args.dry_run:
        print(SYSTEM, "\n\n" + build_prompt(args.question, retriever.search(args.question, k=TOP_K)))
        return

    from llm import BUDGET_USD, spent_usd
    r = answer(args.question, retriever)
    print(r["answer"].strip(), "\n")
    print("Kaynaklar:")
    for c in r["citations"]:
        where = ", ".join(f"Metin {n}" for n in c["chunk_ranks"]) or "getirilen metinlerde yok"
        print(f"  [{c['doc']}, {c['section']}] -> {where}")
    if not r["citations"]:
        print("  (atıf yok)")
    print("\nGetirilen metinler:")
    for c in r["retrieved"]:
        print(f"  {c['rank']}. {c['doc']} | {c['section'] or 'başlık öncesi'} | {', '.join(c['sections'])}")
    print(f"\nToken: {r['input_tokens']} girdi, {r['output_tokens']} çıktı | maliyet: ${r['cost_usd']:.5f} "
          f"| toplam harcama: ${spent_usd():.4f} / ${BUDGET_USD:.2f}")


if __name__ == "__main__":
    main()
