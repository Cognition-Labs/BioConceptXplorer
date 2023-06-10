import sys
import modal
import os
import numpy as np
import time
from sentence_transformers import SentenceTransformer

BCV_list = None
BCV_values = None
BCV_descriptions = None
BERT_sentences = None
BERT_embeddings = None
model = None

DATA_PATH = "/root/embeddings" if not modal.is_local() else "embeddings"


def is_BCV_initialized():
    return (
        BCV_list is not None and BCV_values is not None and BCV_descriptions is not None
    )


def is_BERT_initialized():
    return BERT_sentences is not None and BERT_embeddings is not None


def is_model_initialized():
    return model is not None


def is_everything_initialized():
    return is_BCV_initialized() and is_BERT_initialized() and is_model_initialized()


def init_BCVs():
    global BCV_list, BCV_values, BCV_descriptions, rev_BCV_descriptions
    start = time.time()
    print("- Loading BCVs...", end="")
    BCV_list, BCV_values, BCV_descriptions = (
        np.load(os.path.join(DATA_PATH, "BCV_list.npy")),
        np.load(os.path.join(DATA_PATH, "BCV_values.npy")),
        np.load(
            os.path.join(DATA_PATH, "BCV_descriptions.npy"), allow_pickle=True
        ).item(),
    )
    print(f"Done in {time.time() - start} seconds!")


def init_BERTs():
    global BERT_sentences, BERT_embeddings
    start = time.time()
    print("- Loading BERT embeddings...", end="")
    BERT_embeddings = np.load(os.path.join(DATA_PATH, "BERT_embeddings.npy"))
    print(f"Done in {time.time() - start} seconds!")

    start = time.time()
    print("- Loading BERT sentences...", end="")
    BERT_sentences = np.load(os.path.join(DATA_PATH, "BERT_sentences.npy"))
    print(f"Done in {time.time() - start} seconds!")


def init_model():
    global model
    start = time.time()
    print("- Loading model...", end="")
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    print(f"Done in {time.time() - start} seconds!")


def init_all_if_needed():
    if not is_everything_initialized():
        print("----- Cold Start -----")
        start = time.time()
        init_BCVs()
        init_BERTs()
        init_model()
        print(f"----- Finished initializing in {time.time() - start} seconds! -----")

        print(
            f"""
            BCV_list:        \t{type(BCV_list)}  \t {BCV_list.shape} \t{sys.getsizeof(BCV_list)/1024/1024} MB
            BCV_values:      \t{type(BCV_values)} \t {BCV_values.shape}\t {sys.getsizeof(BCV_values)/1024/1024} MB
            BCV_descriptions:\t{type(BCV_descriptions)} \t\t\t {len(BCV_descriptions)} \t{sys.getsizeof(BCV_descriptions)/1024/1024} MB
            BERT_sentences:  \t{type(BERT_sentences)} \t {BERT_sentences.shape} \t{sys.getsizeof(BERT_sentences)/1024/1024} MB
            BERT_embeddings: \t{type(BERT_embeddings)} \t {BERT_embeddings.shape} \t{sys.getsizeof(BERT_embeddings)/1024/1024} MB
            """
        )
        print("----- End Cold Start -----")
