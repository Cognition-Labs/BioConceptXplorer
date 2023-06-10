import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import modal
from sklearn.metrics.pairwise import cosine_similarity
from core.init import init_all_if_needed
from core.chatgpt import load_openai_key, gpt_rationale, GPTVersion

init_all_if_needed()  # init all the data then load it
from core.init import (
    model,
    BERT_embeddings,
    BERT_sentences,
    BCV_list,
    BCV_values,
    BCV_descriptions,
)

DATA_PATH = "/root/embeddings"
ENV_PATH = "/root/.env"
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

mounts = [
    modal.Mount.from_local_file(".env", remote_path=ENV_PATH),
    modal.Mount.from_local_dir("embeddings", remote_path=DATA_PATH),
]

stub = modal.Stub(name="BioConceptVecXplorer", mounts=mounts, image=image)


@stub.function(cpu=2, memory=2048, container_idle_timeout=300, keep_warm=1)
@modal.web_endpoint(method="GET")
def bert_query(query: str, top_k: int = 10):
    query_vector = model.encode([query])
    sims = cosine_similarity(query_vector, BERT_embeddings)[0]
    indices = np.argsort(sims)[::-1][:top_k]
    options = {
        BERT_sentences[i].strip("\n"): float(sims[j]) for j, i in enumerate(indices)
    }
    return options


@stub.function()
@modal.web_endpoint(method="GET")
def compute_expression(expression: list, top_k: int = 10) -> dict:
    pass


def get_BCV_vector(query: str):
    query_index = np.where(BCV_list == query)[0][0]
    query_vector = BCV_values[query_index]
    return query_vector


def map_BCV_to_description(unmapped: str):
    mapped = BCV_descriptions[unmapped] if unmapped in BCV_descriptions else "N/A"
    if type(mapped) == list:
        mapped = " or ".join(mapped)
    return mapped


"""
Given a single concept query, performs a free variable search using equation of the form
Q + B - C = D
where Q is the query and B, C, D are free variables.
"""


@stub.function(cpu=2, memory=2048, container_idle_timeout=300, keep_warm=1)
@modal.web_endpoint(method="GET")
def free_var_search(
    query: str,
    n: int = 1_000,
    sim_threshold: float = 0.80,
    use_gpt: GPTVersion = GPTVersion.NONE,
):
    q = query.strip("\n")
    if q not in BCV_list:
        return {
            "error": f"Query {q} not found in BioConceptVectors. Please try another query."
        }
    q_mapped = map_BCV_to_description(q)
    print(f"----- Performing free variable search for {q_mapped} ({query})... -----")
    start_freevar = time.time()

    # Get the query vector from BCV embeddings
    q_vector = get_BCV_vector(q)

    # Perform the free variable search
    # - generate n random equations (B, C) pairs by picking indices from 0-->len(BCV_list)
    print(f"- Generating {n} equation samples...", end=" ", flush=True)
    start = time.time()
    equation_indices = np.random.choice(len(BCV_list), size=(n, 2))
    print(f"Done in {time.time() - start} seconds")

    print(f"- Computing B, C, D vectors...", end=" ", flush=True)
    start = time.time()

    # use indices to get B, C vectors
    b_vectors = BCV_values[equation_indices[:, 0]]
    c_vectors = BCV_values[equation_indices[:, 1]]

    # compute D vector
    d_vectors = q_vector.repeat(n).reshape(n, -1) + b_vectors - c_vectors
    print(f"Done in {time.time() - start} seconds")

    # compute cosine similarity between D and all BCVs
    print(f"- Computing cosine similarities...", end=" ", flush=True)
    start = time.time()
    sims = cosine_similarity(d_vectors, BCV_values)
    print(f"Done in {time.time() - start} seconds")

    # get top 4 results
    print(f"- Indexing results...", end=" ", flush=True)
    start = time.time()
    K = 4  # worst case Q, B, C all in top 3 in which case we pick the 4th
    indices = np.argpartition(sims, -K, axis=1)[:, -K:].copy()

    # get indices of b, c, q vectors to remove from top 4 (don't want D = one of B, C, Q)
    b_indices = equation_indices[:, 0]
    c_indices = equation_indices[:, 1]
    q_index = np.where(BCV_list == query)[0][0]
    print(f"Done in {time.time() - start} seconds")

    results = []
    print(f"- Mapping results to build dataframe...", end=" ", flush=True)
    start = time.time()
    for i, (b_idx, c_idx) in tqdm(enumerate(equation_indices), total=n):
        b = BCV_list[b_idx]
        c = BCV_list[c_idx]

        # argpartition doesn't sort the top k, so we sort first
        indices[i] = indices[i][np.argsort(sims[i][indices[i]])][::-1]
        # then calc d index = remove b, c, q indices from top 5 and get the first one
        d_idx = indices[i][
            ~np.isin(
                indices[i], [b_indices[i], c_indices[i], q_index], assume_unique=True
            )
        ][0]
        d = BCV_list[d_idx]
        similarity = sims[i][d_idx]

        if similarity < sim_threshold:
            continue

        # map to descriptions and get equations/similarity
        b_mapped = map_BCV_to_description(b)
        c_mapped = map_BCV_to_description(c)
        d_mapped = map_BCV_to_description(d)
        eq = f"({query}) + ({b}) - ({c}) = ({d})"
        eq_mapped = f"{q} (aka {q_mapped}) + {b} (aka {b_mapped}) - {c} (aka {c_mapped}) = {d} (aka {d_mapped})"

        results.append(
            (
                eq,
                q,
                b,
                c,
                d,
                eq_mapped,
                q_mapped,
                b_mapped,
                c_mapped,
                d_mapped,
                similarity,
            )
        )

    df = pd.DataFrame(
        results,
        columns=[
            "Equation",
            "Q",
            "B",
            "C",
            "D",
            "Equation (Mapped)",
            "Q (Mapped)",
            "B (Mapped)",
            "C (Mapped)",
            "D (Mapped)",
            "Similarity",
        ],
    )
    print(f"Done in {time.time() - start} seconds")

    # sort by similarity
    df = df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)

    if use_gpt != GPTVersion.NONE:
        load_openai_key(ENV_PATH)
        print(
            f"- Getting {use_gpt} rationales for top 1 of {len(df)} rows...",
            end=" ",
            flush=True,
        )
        start = time.time()
        df["Rationale"] = df["Equation (Mapped)"].apply(lambda _: "N/A")
        df.loc[0, "Rationale"] = gpt_rationale(
            df.loc[0, "Equation (Mapped)"], gpt_version=use_gpt
        )

    print(
        f"----- Finished Free Variable Search in {time.time() - start_freevar} seconds -----"
    )
    return df.to_dict(orient="records")


if __name__ == "__main__":
    print(free_var_search("Gene_6853"), file=open("output.csv", "w+"))
