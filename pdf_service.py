from pypdf import PdfReader
from openai import OpenAI
import math


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(pdf_file) -> dict:
    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = page.extract_text() or ""
        clean_text = page_text.strip()

        pages.append(
            {
                "page_number": page_number,
                "text": clean_text,
            }
        )

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    if not full_text:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    return {
        "full_text": full_text,
        "pages": pages,
        "page_count": len(reader.pages),
    }


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def chunk_text(
    text,
    chunk_size,
    overlap,
):
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    i = 0

    while i < len(text):
        chunk = text[
            i:i + chunk_size
        ]

        chunks.append(chunk)

        i += chunk_size - overlap

    return chunks


# --------------------------------------------------
# CHUNK ALL PDF PAGES
# --------------------------------------------------

def chunk_pages(
    pages,
    chunk_size,
    overlap,
):
    chunks = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        # Skip pages where no readable text was extracted
        if not page_text:
            continue

        page_chunks = chunk_text(
            page_text,
            chunk_size,
            overlap,
        )

        for chunk in page_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "page": page_number,
                }
            )

    return chunks


# --------------------------------------------------
# CREATE ONE EMBEDDING
# --------------------------------------------------

def get_embedding(
    text,
    api_key,
):
    client = OpenAI(
        api_key=api_key
    )

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding


# --------------------------------------------------
# ADD EMBEDDINGS TO PDF CHUNKS
# --------------------------------------------------

def add_embeddings_to_chunks(
    chunks,
    api_key,
):
    for chunk in chunks:
        chunk["embedding"] = get_embedding(
            chunk["text"],
            api_key,
        )

    return chunks


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(
    vector_a,
    vector_b,
):
    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            a * a
            for a in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            b * b
            for b in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


# --------------------------------------------------
# FIND MOST RELEVANT PDF CHUNKS
# --------------------------------------------------

def find_relevant_chunks(
    question,
    chunks,
    api_key,
    top_k=3,
):
    if not chunks:
        return []

    question_embedding = get_embedding(
        question,
        api_key,
    )

    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(
            question_embedding,
            chunk["embedding"],
        )

        scored_chunks.append(
            {
                "chunk": chunk,
                "score": score,
            }
        )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]


# --------------------------------------------------
# BUILD CONTEXT FOR THE AI
# --------------------------------------------------

def build_context(best_chunks):
    context_parts = []

    for item in best_chunks:
        chunk = item["chunk"]

        context_parts.append(
            f"Page {chunk['page']}:\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    return context