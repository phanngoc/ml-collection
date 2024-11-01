# Biological Network Analysis: Mathematical Framework and Examples

1. Network Fundamentals
A biological network can be represented as a graph $G = (V, E)$, where:

- $V$ = set of vertices (nodes) representing biological entities
- $E$ = set of edges representing interactions/relationships

**Example Dataset: Protein-Protein Interaction Network**

**Proteins (Nodes $V$):**
- $P1$: Insulin Receptor
- $P2$: IRS1
- $P3$: PI3K
- $P4$: AKT
- $P5$: GLUT4

**Interactions (Edges $E$):**
- $(P1,P2)$: binding
- $(P2,P3)$: phosphorylation
- $(P3,P4)$: activation
- $(P4,P5)$: translocation

2. Network Metrics

### 2.1 Degree Centrality
For a node $v$: $C_D(v) = \frac{\text{deg}(v)}{|V| - 1}$

**Example calculation for $P2$:**

- Degree of $P2$ = 2 (connects to $P1$ and $P3$)
- Total nodes = 5
- $C_D(P2) = \frac{2}{5-1} = 0.5$

### 2.2 Adjacency Matrix
For our example network:

```
    P1  P2  P3  P4  P5
P1   0   1   0   0   0
P2   1   0   1   0   0
P3   0   1   0   1   0
P4   0   0   1   0   1
P5   0   0   0   1   0
```

3. Network Analysis Methods

### 3.1 Clustering Coefficient
For a node $v$ with $k$ neighbors: $CC(v) = \frac{2 \times L}{k \times (k-1)}$

where $L$ is the number of links between neighbors

**Example for $P3$:**

- $k$ = 2 neighbors ($P2$, $P4$)
- $L$ = 0 links between neighbors
- $CC(P3) = \frac{2 \times 0}{2 \times 1} = 0$

### 3.2 Path Analysis
Shortest path lengths:

- $P1 \rightarrow P5$: 4 steps ($P1 \rightarrow P2 \rightarrow P3 \rightarrow P4 \rightarrow P5$)
- Average path length = 2.5

4. Biological Interpretation

**Network Motifs**

Common patterns in our example:

- Linear cascade: $P1 \rightarrow P2 \rightarrow P3 \rightarrow P4 \rightarrow P5$ (Represents insulin signaling pathway)

**Functional Modules**

Identified modules:

- Signal reception ($P1$, $P2$)
- Signal transduction ($P3$, $P4$)
- Response execution ($P5$)

5. Statistical Significance

### 5.1 Random Network Comparison
$Z$-score calculation: $Z = \frac{N_{\text{real}} - \langle N_{\text{rand}} \rangle}{\text{std}(N_{\text{rand}})}$

Where:

- $N_{\text{real}}$ = observed metric
- $\langle N_{\text{rand}} \rangle$ = average in random networks
- $\text{std}(N_{\text{rand}})$ = standard deviation in random networks

**Example:**

For clustering coefficient:

- Observed = 0.15
- Random average = 0.05
- Standard deviation = 0.02
- $Z \text{-score} = \frac{0.15 - 0.05}{0.02} = 5$

6. Applications

**Disease Analysis Example**

Network perturbation analysis:

- Remove node $P3$ (simulating protein inhibition)
- New reachability matrix shows disconnected components
- Pathway disruption score = 0.8

**Drug Target Identification**

Critical nodes identified by centrality:

- $P3$ (betweenness centrality = 0.67)
- $P4$ (betweenness centrality = 0.50)

These represent potential drug targets in the pathway
