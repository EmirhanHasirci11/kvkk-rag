"""Split text into fixed-size word chunks with optional overlap."""


def chunk_words(text: str, size: int, overlap: int = 0) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


if __name__ == "__main__":
    text = " ".join(str(i) for i in range(10))
    print(chunk_words(text, size=4, overlap=0))
    print(chunk_words(text, size=4, overlap=2))
    print(chunk_words("", size=4))
    print(chunk_words("a b", size=4, overlap=1))