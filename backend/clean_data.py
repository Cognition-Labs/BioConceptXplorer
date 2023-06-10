import json
import pickle
import numpy as np


BCVs = json.load(open("unprocessed/concept_glove.json", "r"))
BERT_sentences = np.array(open("unprocessed/sentences.txt", "r").readlines(), dtype=str)
BERT_embeddings = np.load("unprocessed/description_embeddings.npy", "r")
BCV_descriptions = pickle.load(open("unprocessed/concept_descriptions.pkl", "rb"))

temp = {}
for key, value in BCV_descriptions.items():
    if type(value) == str or (type(value) == list and len(value) > 1):
        temp[key] = value
    elif type(value) == list and len(value) == 1:
        temp[key] = value[0]

BCV_descriptions = temp
rev_BCV_descriptions = {}

for key, value in BCV_descriptions.items():
    if type(value) == str:
        rev_BCV_descriptions[value] = key
    else:
        for v in value:
            rev_BCV_descriptions[v] = key


BCV_list = np.array(list(BCVs.keys()), dtype=str)
BCV_values = np.array(list(BCVs.values()), dtype=np.float32)
BCVs = {BCV_list[i]: np.array(BCV_values[i]) for i in range(len(BCV_list))}

print(f"BCVs: {len(BCVs)}")
print(f"BCV_list: {BCV_list.shape}")
print(f"BCV_values: {BCV_values.shape}")
print(f"BERT_sentences: {BERT_sentences.shape}")
print(f"BERT_embeddings: {BERT_embeddings.shape}")
print(f"BCV_descriptions: {len(BCV_descriptions)}")
print(f"rev_BCV_descriptions: {len(rev_BCV_descriptions)}")


np.save("embeddings/BCVs.npy", BCVs)
np.save("embeddings/BCV_values.npy", BCV_values)
np.save("embeddings/BERT_embeddings.npy", BERT_embeddings)
np.save("embeddings/BCV_list.npy", BCV_list)
np.save("embeddings/BERT_sentences.npy", BERT_sentences)
np.save("embeddings/BCV_descriptions.npy", BCV_descriptions)
np.save("embeddings/rev_BCV_descriptions.npy", rev_BCV_descriptions)
