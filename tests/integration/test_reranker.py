import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = CrossEncoder(
        MODEL_NAME,
        device=device,
        max_length=512
    )

    query = "What is cgroup v2?"

    passages = [
        (
            "cgroup v2 is the next version of the Linux "
            "`cgroup` API. cgroup v2 provides a unified "
            "control system with enhanced resource management "
            "capabilities."
        ),
        (
            "The recommended way to use cgroup v2 is to use "
            "a Linux distribution that enables and uses "
            "cgroup v2 by default."
        ),
        (
            "The cgroup version depends on the Linux distribution "
            "being used and the default cgroup version configured "
            "on the OS."
        ),
        (
            "The kubelet supports manipulation of the cgroup "
            "hierarchy on the host using a cgroup driver."
        )
    ]

    pairs = [
        [query, passage]
        for passage in passages
    ]

    scores = model.predict(
        pairs,
        batch_size=4,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(scores, passages),
        key=lambda x: x[0],
        reverse=True
    )

    print("\nReranker results:")
    print("=" * 80)

    for rank, (score, passage) in enumerate(
        ranked,
        start=1
    ):

        print(
            f"\n[{rank}] Score: {float(score):.4f}"
        )

        print(passage)


if __name__ == "__main__":
    main()