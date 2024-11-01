# Locality-Sensitive Hashing (LSH) 

is a technique used to efficiently find approximate nearest neighbors in high-dimensional spaces. It hashes input items so that similar items are more likely to be hashed to the same bucket, reducing the number of comparisons needed. LSH is especially useful in tasks like nearest neighbor search, clustering, or duplicate detection, where exact matching is computationally expensive.
```markdown
## Example Scenario: Finding Similar Documents

Let's say we have a set of text documents, and we want to quickly find documents similar to a given one. We’ll use LSH to speed up this process by focusing on similarities in the documents' content.

### Example Data

Assume we have the following documents:

```
Doc 1: "Apple is a technology company based in the US."
Doc 2: "Microsoft is a leading software company in the US."
Doc 3: "Apple and Microsoft are major technology firms."
```

### Step 1: Vectorize Documents

First, we convert these documents into vectors. A simple approach could be using TF-IDF or binary vectors based on the presence of words. For simplicity, let's say we create a binary vector for each document:

```
Doc 1: [1, 0, 1, 1, 0, 1]
Doc 2: [0, 1, 1, 0, 1, 1]
Doc 3: [1, 1, 1, 1, 0, 1]
```

Here, each position in the vector represents the presence (1) or absence (0) of a specific term.

### Step 2: Define a Hash Function

LSH uses hash functions designed to maximize the probability that similar documents will fall into the same bucket. For simplicity, let’s assume a hash function that takes a random subset of the vector components:

```
Hash Function 1: Takes the 1st and 3rd components of the vector and concatenates them.
```

Applying this hash function:

```
Doc 1: [1, 1] → Hash = 11
Doc 2: [0, 1] → Hash = 01
Doc 3: [1, 1] → Hash = 11
```

### Step 3: Bucket Documents Based on Hashes

We group documents that have the same hash value into the same bucket:

```
Bucket 11: Doc 1, Doc 3
Bucket 01: Doc 2
```

This means that if we query for documents similar to Doc 1, we only need to check Doc 3 (since they share the same hash), significantly reducing the number of comparisons.

### Step 4: Evaluate Similarity

Once documents are grouped in the same bucket, we compute the actual similarity (e.g., cosine similarity) only between documents within that bucket.

### Result

By using LSH, we quickly narrowed down the candidates for similarity checking, saving computation time.

This is a simple example, but LSH is highly customizable with multiple hash functions and hashing layers to achieve accurate results, even for complex data like images or large text corpora.
```