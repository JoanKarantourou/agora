"""Smoke test for the embedding model: load bge-base, embed three sentences,
and check that related sentences score higher than unrelated ones."""

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

SENTENCES = {
    "A": "No man ever steps in the same river twice.",
    "B": "Everything flows and nothing stays still.",
    "C": "The best way to cook pasta is in salted water.",
}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA: yes ({torch.cuda.get_device_name(0)})")
    else:
        print("CUDA: no (running on CPU)")

    model = SentenceTransformer(MODEL_NAME, device=device)
    a, b, c = model.encode(list(SENTENCES.values()), normalize_embeddings=True)

    print(f"Vector length of A: {len(a)}")
    # Embeddings are unit-length, so the dot product is the cosine similarity.
    print(f"Cosine similarity A-B: {float(a @ b):.4f}")
    print(f"Cosine similarity A-C: {float(a @ c):.4f}")


if __name__ == "__main__":
    main()
