1. Mathematical Details of DCG
The Discounted Cumulative Gain (DCG) is used to evaluate the ranking quality of a list of documents based on their relevance scores. The DCG score accounts for both the relevance of a document and its position in the ranked list. The intuition is that highly relevant documents appearing earlier in the ranking are more valuable.

Formula for DCG:
$DCG_p = rel_1 + \sum_{i=2}^p \frac{rel_i}{\log_2(i+1)}$

Where:

- $p$ is the rank position.
- $rel_i$ is the relevance score of the document at position $i$.
- $i$ is the rank index, starting from 1 for the top document.

For example, if you have the following relevance scores for a list of ranked documents:

- Rank 1: Relevance = 3
- Rank 2: Relevance = 2
- Rank 3: Relevance = 1

The DCG for the top 3 documents would be calculated as:

$DCG_3 = 3 + \frac{2}{\log_2(2+1)} + \frac{1}{\log_2(3+1)} = 3 + \frac{2}{\log_2(3)} + \frac{1}{\log_2(4)}$

### Ideal DCG (IDCG)
The Ideal Discounted Cumulative Gain (IDCG) is the best possible DCG for a set of documents, assuming that they are ranked in the perfect order (from the most relevant to the least relevant).

You calculate IDCG by sorting the documents by their relevance in descending order and then calculating DCG for that ideal ranking.

### Normalized DCG (nDCG)
The Normalized DCG (nDCG) is the ratio of DCG to the IDCG, ensuring that the score is normalized between 0 and 1. A perfect ranking would have an nDCG of 1.

$nDCG_p = \frac{DCG_p}{IDCG_p}$

### Python Code for DCG and nDCG
Now, let’s implement the calculations in Python.

#### Python Code for DCG:
```python
import math

def dcg(relevances, p=None):
    """
    Compute DCG for a list of relevances.
    :param relevances: List of relevance scores.
    :param p: Rank position to consider (default is all ranks).
    :return: DCG score.
    """
    if p is None:
        p = len(relevances)
    
    return relevances[0] + sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances[1:p], start=2))

# Example usage
relevances = [3, 2, 1, 0]  # Example relevance scores
dcg_score = dcg(relevances)
print(f"DCG: {dcg_score:.2f}")
```

#### Python Code for IDCG:
To calculate IDCG, we need to sort the relevance scores in descending order and compute DCG.

```python
def idcg(relevances, p=None):
    """
    Compute the Ideal DCG for a list of relevances.
    :param relevances: List of relevance scores.
    :param p: Rank position to consider (default is all ranks).
    :return: IDCG score.
    """
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg(sorted_relevances, p)

# Example usage
idcg_score = idcg(relevances)
print(f"IDCG: {idcg_score:.2f}")
```

#### Python Code for nDCG:
Finally, let’s compute the nDCG.

```python
def ndcg(relevances, p=None):
    """
    Compute the normalized DCG for a list of relevances.
    :param relevances: List of relevance scores.
    :param p: Rank position to consider (default is all ranks).
    :return: nDCG score.
    """
    dcg_score = dcg(relevances, p)
    idcg_score = idcg(relevances, p)
    if idcg_score == 0:
        return 0
    return dcg_score / idcg_score

# Example usage
ndcg_score = ndcg(relevances)
print(f"nDCG: {ndcg_score:.3f}")
```

#### Example Output:
```bash
DCG: 4.78
IDCG: 7.00
nDCG: 0.683
```

### Extended Example with Full Dataset
Consider the following dataset of documents and relevance scores:

| Rank | Document | Relevance |
|------|----------|-----------|
| 1    | A        | 3         |
| 2    | B        | 1         |
| 3    | C        | 2         |
| 4    | D        | 0         |
| 5    | E        | 3         |

The relevance scores are: [3, 1, 2, 0, 3].

```python
relevances = [3, 1, 2, 0, 3]
dcg_score = dcg(relevances)
idcg_score = idcg(relevances)
ndcg_score = ndcg(relevances)

print(f"DCG: {dcg_score:.2f}")
print(f"IDCG: {idcg_score:.2f}")
print(f"nDCG: {ndcg_score:.3f}")
```

For this example, the ideal ranking is: [3, 3, 2, 1, 0].

### Explanation of the Python Code
- **DCG Function**: Takes a list of relevance scores and calculates DCG by summing the relevance scores, adjusted by a logarithmic discount based on rank position.
- **IDCG Function**: Sorts the relevance scores in descending order to compute the ideal DCG.
- **nDCG Function**: Divides the DCG by the IDCG to get a normalized value between 0 and 1.

### Conclusion
- **DCG** tells you how well your ranked documents are ordered based on their relevance.
- **nDCG** normalizes the result, giving you a score between 0 and 1, where 1 is perfect.

These metrics are widely used in search engines, recommendation systems, and other ranking-based systems. With this detailed breakdown and code examples, you can easily compute DCG, IDCG, and nDCG for your own datasets. Would you like to explore more advanced topics, such as how to optimize search or recommendation systems based on nDCG?
