from itertools import islice
import time
import os
import pandas as pd
import shelve
import json
import numpy as np
import pickle
from tqdm import tqdm
import modal


DATA_PATH = "/root/embeddings"
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

mounts = [
    modal.Mount.from_local_file(".env", remote_path="/root/.env"),
    modal.Mount.from_local_directory("embeddings", remote_path=DATA_PATH),
]

stub = modal.Stub(name="bioconceptexplorer", mounts=mounts, image=image)

def load_embeddings_data():
    print("Cold start\n- loading embeddings from shared volume cache...")
    with open(os.path.join(DATA_PATH, "concept_glove.json")) as json_file:
        concept_glove = json.load(json_file)
        print("loaded embeddings file")
        concept_list = list(concept_glove.keys())
        print("loaded concept list")
        embedding_values = np.array(list(concept_glove.values()), dtype=np.float32)
        print("loaded embedding values")
    return concept_glove, concept_list, embedding_values

def load_bert_embeddings():
    sentence_embeddings = np.load(os.path.join(DATA_PATH, "description_embeddings.npy"))
    return sentence_embeddings

def load_bert_sentences():
    with open(os.path.join(DATA_PATH, "sentences.txt")) as f:
        sentences = f.readlines()
    return sentences

concept_glove, concept_list, embedding_values = load_embeddings_data()
sentence_embeddings = load_bert_embeddings()
sentences = load_bert_sentences()

@stub.function()
@modal.web_endpoint(method="GET")
def bert_query(query: str, top_k: int = 10):
    similarities = np.dot(sentence_embeddings, sentence_embeddings[concept_list.index(query)])
    top_k_idx = np.argsort(similarities)[::-1][:top_k]
    return [sentences[idx] for idx in top_k_idx]

@stub.function()
@modal.web_endpoint(method="GET")
def compute_expression(expression: list, top_k: int = 10) -> dict:
    pass

@stub.function()
@modal.web_endpoint(method="GET")
def free_var_search(query: str, top_k: int = 10):
    pass

@stub.function()
@modal.web_endpoint(method="GET")
def heartbeat():
    return "OK"