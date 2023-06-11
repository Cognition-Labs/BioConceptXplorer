# BioConceptVecXplorer

Using literature-based (PubMed) concept embeddings to explore and test biological hypotheses at scale.

Streamlit Batch Processing Demo: https://shreyj1729-bioconceptxplorer-streamlitmain-50qmd1.streamlit.app/

## Setting up the dev environment

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

The `venv` directory and the `requirements.txt` file are in the root directory of the folder so that vs-code selects them automatically on opening the project folder.

Future Direction:

- Pick concept in template (species, protein, mutation, disease)
- Pick negative threshold range (too close to input or some other cluster)
- cluster results based on context
- - send members to chatgpt to generate cluster headings
- - interactive clustering with gpt-4 to label, visual PCA or t-SNE

https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/

- TLDR embeddings represent diction not semantic meaning
- use logarithm strategy mentioned in above article to get more meaningful results

maybe finetune Instructor Embeddings on PubMed corpus?

Also SciHub is a thing, has >60M bio/med/chem papers
