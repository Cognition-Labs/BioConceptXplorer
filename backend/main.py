import time
import os
import pandas as pd
import shelve
import json
import numpy as np
import pickle
from tqdm import tqdm
import modal

# shared volume for concept embeddings and mapping
volume = modal.SharedVolume().persist("embeddings-cache-vol")
CACHE_PATH = "/root/cache/"
shared_volumes = {CACHE_PATH: volume}

# mounts for env file
mounts = [
    modal.Mount.from_local_file(".env", remote_path="/root/.env"),
    modal.Mount.from_local_file("embeddings/concept_glove.json", remote_path="/root/cache/concept_glove.json"),
]

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

stub = modal.Stub(name="bioconceptexplorer", mounts=mounts, image=image)
stub.concept_embeddings = modal.Dict()
stub.concept_list = modal.Dict()
stub.embedding_values = modal.Dict()
stub.concept_descriptions = modal.Dict()
stub.rev_concept_descriptions = modal.Dict()

@stub.function(shared_volumes=shared_volumes)
def load_embeddings_data():
    print("Running locally: ", modal.is_local())
    print("Cold start\n- loading embeddings from shared volume cache...")
    embeddings = open("/root/cache/concept_glove.json").readlines()
    print(len(embeddings))

    with open("/root/cache/concept_glove.json") as json_file:
        stub.concept_embeddings = json.load(json_file)
        print("loaded embeddings file")
        stub.concept_list = {"data": list(stub.concept_embeddings.keys())}
        print("loaded concept list")
        stub.embedding_values = {"data": np.array(list(stub.concept_embeddings.values()), dtype=np.float32)}
        print("loaded embedding values")


@stub.function()
@modal.web_endpoint(method="GET")
def heartbeat():
    print([i for i in os.walk("/root")])
    print(open("/root/.env").read())
    embeddings = open("/root/cache/concept_glove.json").readlines()
    print(len(embeddings))
    return "OK"