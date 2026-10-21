import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        }
    )

    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    texts = [
        "What is Kubernetes?",
        "Kubernetes is a container orchestration platform.",
        "How does kubelet communicate with the API server?"
    ]

    embeddings = model.encode(
        texts,
        batch_size=3,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"First vector first 5 values: {embeddings[0][:5]}")

    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            f"GPU memory reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )


if __name__ == "__main__":
    main()