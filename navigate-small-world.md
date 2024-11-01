# Navigable Small World

Let's analyze the NSW approach with concrete examples and calculations:

### Graph Construction Example:
Consider 5 vectors in 3D space:

```python
v0 = [0.1, 0.2, 0.3]
v1 = [0.2, 0.3, 0.4]
v2 = [0.7, 0.8, 0.9]
v3 = [0.3, 0.4, 0.5]
v4 = [0.8, 0.7, 0.6]
```

### Distance Calculations:
For vectors `v0` and `v1`:

```python
distance = √[(0.2-0.1)² + (0.3-0.2)² + (0.4-0.3)²]
         = √[0.01 + 0.01 + 0.01]
         = √0.03
         ≈ 0.173
```

### Connection Types:
For node `v0`:

**Short-range connections (k=2):**
- To `v1` (distance: 0.173)
- To `v3` (distance: 0.346)

**Long-range connection (k=1):**
- To `v2` (distance: 1.039)

### Search Process Example:
Query `q = [0.5, 0.5, 0.5]`

```python
# Step 1: Start at random entry point v0
# Distance to q: 0.520

# Step 2: Explore neighbors of v0
# v1: distance = 0.424
# v2: distance = 0.519
# v3: distance = 0.173

# Step 3: Move to closest unexplored neighbor (v3)
# Explore its neighbors...
```

### Complexity Analysis:
For `n` nodes:

- **Construction time:** `O(n log n)`
- **Search time:** `O(log n)` expected
- **Space complexity:** `O(n k)` where `k = short_range_k + long_range_k`

### Performance Metrics:
Example dataset (`n=1000`)

```python
# Average degree = short_range_k + long_range_k = 5 + 2 = 7
# Average search steps = O(log 1000) ≈ 10 steps
```

### Probabilistic Properties:
The probability of having a long-range connection at distance `d`:

```python
P(d) ∝ 1/d^r
# where r is typically chosen around 2 for optimal performance.
```

### Search Quality Metrics:
For a typical search with `ef=10`:

```python
# Recall@1: ~0.95
# Recall@10: ~0.90
# Average query time: ~O(log n)
```

### Key Advantages of NSW:

**Balanced Structure:**

- Short-range links ensure local connectivity
- Long-range links provide "shortcuts" for faster navigation
- Natural clustering emerges from distance-based connections

**Efficient Search:**

- Greedy search strategy works well due to graph structure
- Logarithmic complexity in practice
- Good balance between accuracy and speed

**Adaptability:**

- Works well with different distance metrics
- Can be adapted for various data types
- Scales well with dimensionality

The combination of short-range and long-range connections creates a "small world" effect, where:

- Most nodes are not neighbors
- But most nodes can be reached through a small number of hops
- Search can efficiently navigate from distant to nearby points

This structure provides an excellent balance between:

- Construction complexity
- Search efficiency
- Memory usage
- Search accuracy

The implementation shows how NSW achieves fast approximate nearest neighbor search through:

- Intelligent graph construction
- Balanced connection strategy
- Efficient greedy search
- Probabilistic guarantees

This makes NSW particularly effective for high-dimensional vector search applications where exact search would be prohibitively expensive.