# Key differences between dense and sparse retrieval in information retrieval systems.

## Dense Retrieval

**Description:**
- Uses dense vector representations (embeddings) of text, typically 256-1024 dimensions.
- Documents and queries are mapped to continuous vectors using neural networks.
- Similarity is measured via vector operations (usually cosine similarity).
- Captures semantic relationships and contextual meaning.


**Examples:**
- BERT-based retrievers
- DPR (Dense Passage Retriever)

**Example Code:**

Here is an example of how to use a BERT-based retriever with the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Encode a query and a document
query = "What is dense retrieval?"
document = "Dense retrieval uses dense vector representations of text."

query_tokens = tokenizer(query, return_tensors='pt')
document_tokens = tokenizer(document, return_tensors='pt')

# Get embeddings
query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1)
document_embedding = model(**document_tokens).last_hidden_state.mean(dim=1)

# Compute cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(query_embedding, document_embedding)

print(f"Cosine Similarity: {cosine_similarity.item()}")
```

And here is an example of using Dense Passage Retriever (DPR) with the `transformers` library:

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch

# Load pre-trained DPR model and tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Encode a query and a document
query = "What is dense retrieval?"
document = "Dense retrieval uses dense vector representations of text."

query_tokens = question_tokenizer(query, return_tensors='pt')
document_tokens = context_tokenizer(document, return_tensors='pt')

# Get embeddings
query_embedding = question_encoder(**query_tokens).pooler_output
document_embedding = context_encoder(**document_tokens).pooler_output

# Compute cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(query_embedding, document_embedding)

print(f"Cosine Similarity: {cosine_similarity.item()}")
```


**Pros:**
- Better at handling semantic similarity and paraphrasing.
- Can find relevant results even with different terminology.
- More compact representations.

**Cons:**
- Computationally intensive to train.
- Requires significant training data.
- Vector similarity search can be expensive.

## Sparse Retrieval

**Description:**
- Uses sparse vector representations (typically bag-of-words).
- Documents and queries represented as high-dimensional sparse vectors.
- Similarity often measured via TF-IDF or BM25.
- Focuses on lexical matching of terms.

**Examples:**
- BM25
- TF-IDF based search

**Example Code:**

Here is an example of how to use BM25 with the `rank_bm25` library:

```python
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Sample documents
documents = [
    "Sparse retrieval uses sparse vector representations.",
    "BM25 is a popular algorithm for sparse retrieval.",
    "TF-IDF is another method for sparse retrieval."
]

# Tokenize the documents
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

# Initialize BM25
bm25 = BM25Okapi(tokenized_documents)

# Query
query = "What is sparse retrieval?"
tokenized_query = word_tokenize(query.lower())

# Get scores
scores = bm25.get_scores(tokenized_query)

print(f"BM25 Scores: {scores}")
```

And here is an example of using TF-IDF with the `sklearn` library:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "Sparse retrieval uses sparse vector representations.",
    "BM25 is a popular algorithm for sparse retrieval.",
    "TF-IDF is another method for sparse retrieval."
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Query
query = "What is sparse retrieval?"
query_vec = vectorizer.transform([query])

# Compute cosine similarity
cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

print(f"TF-IDF Cosine Similarities: {cosine_similarities}")
```


**Pros:**
- Simple and interpretable.
- Works well with exact keyword matches.
- Fast retrieval with inverted indices.
- Requires no training.

**Cons:**
- Misses semantic relationships.
- Struggles with synonyms and paraphrases.
- Storage can be inefficient due to high dimensionality.

Would you like me to elaborate on any specific aspect of these approaches?