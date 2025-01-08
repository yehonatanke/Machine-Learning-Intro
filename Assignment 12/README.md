## (Q1) - Problem 3

Consider the linear regression problem and recall that the least-squares estimate $\hat{w}_{\text{LS}}$ satisfies the normal equations, i.e.,  
```math
X^T X w = X^T y \quad (*)
```
When $X$ has a full column rank, the unique solution is given by $\hat{w}_{\text{LS}} = (X^T X)^{-1} X^T y$.  
When $X$ is not full rank, $X^T X$ is singular and one needs to address the question of whether the normal equations $(*)$ have a solution, and if so, whether it is unique.  

To formalize this observation, we define the range space of $X$, denoted by $\mathcal{R}(X)$, as the linear space spanned by the columns of $X$, i.e.,  
```math
\mathcal{R}(X) = \{Xa | a \in \mathbb{R}^{d+1} \}.
```
We define also the nullspace of $X$, denoted by $\mathcal{N}(X)$, as  
```math
\mathcal{N}(X) = \{a \in \mathbb{R}^{d+1} | Xa = 0 \}.
```

### Prove the following claims:  
#### (a)

```math
\mathcal{N}(X) = \mathcal{R}(X^T)^\perp, \quad \mathcal{N}(X^T) = \mathcal{R}(X)^\perp, \quad \mathcal{R}(X) = \mathcal{N}(X^T)^\perp, \quad \mathcal{R}(X^T) = \mathcal{N}(X)^\perp
```

where $\perp$ denotes the orthogonal complement space of a linear space, defined as  

```math
\mathcal{L} \subset \mathbb{R}^N, \quad \mathcal{L}^\perp = \{a \in \mathbb{R}^N | a^T b = 0, \, \text{for all } b \in \mathcal{L} \}.
```

#### (b)  
```math
\mathcal{R}(X^T X) = \mathcal{R}(X^T)
```

#### (c) 
When $X$ is not full rank, the normal equations $(*)$ always have more than one solution, where any two solutions $\hat{w}_1$ and $\hat{w}_2$ differ by a vector in the nullspace of $X$, i.e.,  
```math
X (\hat{w}_1 - \hat{w}_2) = 0.
```

#### (d) 
The projection of $y$ onto $\mathcal{R}(X)$ is unique and is defined by  

```math
\hat{y} = X \hat{w},
```

where $\hat{w}$ is any solution to the normal equations $(*)$.

#### (e) 
When $X$ has a full column rank, we can write  

```math
\hat{y} = (X^T X)^{-1} X^T y.
```

---

## (Q2) - Problem 4

Recall that the least squares estimate of $w$ is given by $\hat{w}_{LS} = X^{\dagger} y$, 

where $X^{\dagger} = (X^T X)^{-1} X^T$, and that $\hat{y} = X \hat{w}_{LS}$ ($X$ is a $N$ by $d+1$ matrix). 

We define the **hat matrix** $H$ as follows:

```math
H = X (X^T X)^{-1} X^T,
```

such that $\hat{y} = H y$. Show the following properties of $H$:

- Every eigenvalue of $H$ is either 0 or 1.
- How many eigenvalues of $H$ are 1? What is the rank of $H$? 
