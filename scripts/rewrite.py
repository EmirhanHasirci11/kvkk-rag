"""Query rewriting: turn an everyday question into the wording of the KVKK texts before retrieval.

The rewrite is only used as a search query, never shown as an answer. The prompt has no
examples taken from eval/golden.jsonl, only general term mappings.
"""
from llm import generate

SYSTEM = """Görevin, kullanıcının kişisel verilerin korunmasıyla ilgili sorusunu, 6698 sayılı Kişisel Verilerin Korunması Kanunu, ilgili yönetmelik, tebliğ ve Kurum rehberlerinde kullanılan hukuki terimlerle yeniden yazmak. Yazdığın metin bir arama motorunda sorgu olarak kullanılacak.

Kurallar:
1. Soruyu cevaplama, sadece yeniden yaz.
2. Soruda olmayan sayı, süre, tutar, madde numarası veya bilgi ekleme.
3. Gündelik ifadeleri mevzuattaki karşılıklarına çevir; örneğin "müşteri" veya "kullanıcı" yerine "ilgili kişi", "şirket" yerine "veri sorumlusu", "bilgi" yerine "kişisel veri".
4. Sorunun anlamını koru; konuyu daraltma veya genişletme.
5. En fazla iki cümle yaz ve yalnızca yeniden yazılmış sorguyu döndür."""


def rewrite(question: str) -> dict:
    """{"text", "input_tokens", "output_tokens", "cost_usd"} with text = the rewritten query."""
    out = generate(question, SYSTEM)
    return {**out, "text": " ".join(out["text"].split())}
