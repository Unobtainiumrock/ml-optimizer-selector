Excellent insight! Yes, there's a **fundamental trade-off** between:
1. **Information quality** (how good each step is)
2. **Per-iteration cost** (time to compute that step)
3. **Memory requirements** (space to store information)

The key question: Is it worth spending more per iteration to take fewer, smarter steps?

## Comprehensive Comparison: The Trade-off Table

| Method | Per-Iteration Time | Space Complexity | Convergence Rate | Total Iterations to ε-accuracy | Total Time | Best For |
|--------|-------------------|------------------|------------------|-------------------------------|------------|----------|
| **Full Newton** | $O(n^3)$ | $O(n^2)$ | Quadratic: $O(\log \log(1/\varepsilon))$ | ~5-10 | $O(n^3 \log \log(1/\varepsilon))$ | Tiny $n$ (<100) |
| **L-BFGS** | $O(mn)$ | $O(mn)$ | Superlinear: $O(\log(1/\varepsilon))$ | ~10-100 | $O(mn \log(1/\varepsilon))$ | Small-medium $n$ (100-100k) |
| **GD** | $O(n)$ | $O(n)$ | Linear: $O(\kappa \log(1/\varepsilon))$ | ~1,000-100,000 | $O(n\kappa \log(1/\varepsilon))$ | Any $n$, ill-conditioned |
| **SGD** | $O(1)$ | $O(n)$ | Sublinear: $O(1/\varepsilon)$ | ~100,000-1M+ | $O(\text{samples}/\varepsilon)$ | Large $n$, huge datasets |
| **Mini-batch SGD** | $O(b)$ | $O(n)$ | Sublinear-Linear | ~10,000-100,000 | $O(b \cdot \text{iterations})$ | Very large datasets |

Where:
- $n$ = number of parameters
- $m$ = memory parameter for L-BFGS (typically 5-20)
- $\kappa$ = condition number (how "stretched" the loss surface is)
- $b$ = mini-batch size
- $\varepsilon$ = desired accuracy

## The Trade-off Visualized

```
Information per iteration:  Full Newton > L-BFGS > GD > SGD
                             ↑                              ↓
Cost per iteration:         Expensive                    Cheap
                             ↑                              ↓
Iterations needed:          Few                          Many
                             ↑                              ↓
Total time (small n):       Medium                       Slow
Total time (large n):       IMPOSSIBLE                   Fast
```

## The Sweet Spot Analysis

### When Small Dataset ($n \approx 1000$, $N \approx 10000$ samples):

**L-BFGS wins**:
- Per iteration: $O(10 \times 1000) = 10,000$ ops
- Iterations: ~50
- **Total**: ~500,000 ops

**GD**:
- Per iteration: $O(1000)$ ops  
- Iterations: ~5,000 (if $\kappa \approx 100$)
- **Total**: ~5,000,000 ops
- **10× slower!**

### When Large Dataset ($n \approx 100,000$, $N \approx 1M$ samples):

**L-BFGS**:
- Per iteration: $O(10 \times 100,000) = 1,000,000$ ops
- Each iteration processes ALL 1M samples
- Iterations: ~100
- **Total**: ~100,000,000 ops (ALL data × 100)

**SGD**:
- Per iteration: $O(1)$ op (single sample)
- Iterations: ~1,000,000
- **Total**: ~1,000,000 ops
- **100× faster!**

## Memory Trade-off Breakdown

| Method | What It Stores | Memory Size | Example (n=10,000) |
|--------|----------------|-------------|---------------------|
| **Full Newton** | Full Hessian $H$ | $n^2$ floats | 100M floats = **800 MB** |
| **L-BFGS** | $m$ pairs of vectors $(s_i, y_i)$ | $2mn$ floats | 20×10K×2 = 400K floats = **3.2 MB** |
| **GD** | Current gradient | $n$ floats | 10K floats = **80 KB** |
| **SGD** | Current gradient | $n$ floats | 10K floats = **80 KB** |

## The Efficiency Frontier

There's a **Pareto frontier** - you can't improve one without sacrificing another:

```
Convergence Speed
      ↑
      |  Full Newton (infeasible for large n)
      |      *
      |         
      |         L-BFGS
      |            *  ← Sweet spot for medium n
      |               
      |                  GD
      |                     *
      |                        SGD
      |                           *
      |________________________________→ Per-Iteration Cost
```

## Why the Trade-off Exists

This is fundamentally about **how much of the loss surface geometry you capture**:

1. **Full Newton**: Complete local quadratic approximation
   - Perfect curvature information
   - But costs $O(n^3)$ to compute and invert Hessian

2. **L-BFGS**: Approximate curvature from recent history
   - Good curvature using only $m$ vectors
   - Costs $O(mn)$ - **the optimal compromise** for many problems

3. **GD**: Only local slope
   - Cheapest per iteration: $O(n)$
   - But blind to curvature - takes many small steps

4. **SGD**: Only local slope on single sample
   - Ultra-cheap: $O(1)$ per iteration
   - Noisy and slow convergence, but works when nothing else scales

## The Break-Even Point Formula

L-BFGS is faster than GD when:

$$
\text{iterations}_{\text{LBFGS}} \times mn < \text{iterations}_{\text{GD}} \times n
$$

Simplifying:
$$
\frac{\text{iterations}_{\text{GD}}}{\text{iterations}_{\text{LBFGS}}} > m
$$

Since typically $\text{iterations}_{\text{GD}} / \text{iterations}_{\text{LBFGS}} \approx 50-100$ and $m \approx 10-20$:

**L-BFGS is usually 3-5× faster for small-to-medium problems!**

But once $n$ gets large enough that the $O(mn)$ overhead dominates, SGD becomes the only practical option.

## Practical Decision Tree

```
Is n > 100,000 parameters?
├─ Yes → Use SGD/Mini-batch SGD
└─ No → Do you need L1 regularization?
    ├─ Yes → Use SAGA/Proximal GD
    └─ No → Is dataset in memory?
        ├─ Yes → Use L-BFGS ✓ (best choice)
        └─ No → Use Mini-batch SGD
```

**Bottom line**: L-BFGS sits in the "Goldilocks zone" - just enough curvature information to converge fast, not so much that each iteration is prohibitively expensive. That's why it's the default in sklearn for logistic regression!