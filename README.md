# BioConceptVecXplorer

Using literature-based (PubMed) concept embeddings to explore and test biological hypotheses at scale.

- Pick concept in template (species, protein, mutation, disease)
- Pick negative threshold range (too close to input or some other cluster)
- cluster results based on context
- - send members to chatgpt to generate cluster headings
- - interactive clustering with gpt-4 to label, visual PCA or t-SNE

https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/

- TLDR embeddings represent diction not semantic meaning
- use logarithm strategy mentioned in above article to get more meaningful results

maybe finetune Instructor Embeddings on PubMed corpus?
