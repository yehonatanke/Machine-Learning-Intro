# Linear Algebra and Multivariable Calculus Summary

<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#vector-spaces-and-advanced-subspaces">Vector Spaces and Advanced Subspaces</a></li>
    <li><a href="#advanced-linear-transformations">Advanced Linear Transformations</a></li>
    <li><a href="#inner-product-spaces">Inner Product Spaces</a></li>
    <li><a href="#eigendecomposition">Eigendecomposition</a></li>
    <li><a href="#jordan-forms">Jordan Forms</a></li>
    <li><a href="#singular-value-decomposition-svd">Singular Value Decomposition (SVD)</a></li>
    <li><a href="#matrix-factorizations">Matrix Factorizations</a></li>
    <li><a href="#gradient-divergence-curl">Gradient, Divergence, Curl</a></li>
    <li><a href="#jacobian-matrices-and-determinants">Jacobian Matrices and Determinants</a></li>
    <li><a href="#hessian-matrices">Hessian Matrices</a></li>
    <li><a href="#multiple-integrals-and-change-of-variables">Multiple Integrals and Change of Variables</a></li>
    <li><a href="#line-and-surface-integrals">Line and Surface Integrals</a></li>
    <li><a href="#theorems-green-stokes-gauss">Theorems (Green, Stokes, Gauss)</a></li>
    <li><a href="#connections-between-topics">Connections Between Topics</a></li>
  </ul>
</details>

## Vector Spaces and Advanced Subspaces
- **Definition**: A vector space is a set of vectors closed under addition and scalar multiplication.
- **Subspaces**: A subset $W$ of a vector space $V$ is a subspace if $0 \in W$, and $u + v \in W$ and $cu \in W$ for all $u, v \in W$ and $c \in \mathbb{R}$.

**Example**:

$$
W = \{(x, y, z) \in \mathbb{R}^3 : x + y + z = 0\}
$$

is a subspace of $\mathbb{R}^3$.



## Advanced Linear Transformations
- **Definition**: A linear transformation $T: V \to W$ satisfies $T(u + v) = T(u) + T(v)$ and $T(cu) = cT(u)$.
- **Kernel and Image**:
  - $\text{Ker}(T) = \{v \in V : T(v) = 0\}$
  - $\text{Im}(T) = \{T(v) : v \in V\}$

**Example**:

$T(x, y) = (x + y, x - y)$ has $\text{Ker}(T) = \{(0, 0)\}$ and $\text{Im}(T) = \mathbb{R}^2$.



## Inner Product Spaces
- **Definition**: An inner product space is a vector space with an additional operation $\langle u, v \rangle$ satisfying:
  1. Linearity in the first argument
  2. Symmetry: $\langle u, v \rangle = \langle v, u \rangle$
  3. Positive definiteness: $\langle v, v \rangle > 0$ for $v \neq 0$

**Example**:
For $\mathbb{R}^n$:

$$
\langle u, v \rangle = u^T v
$$

is an inner product.


## Inner Product Spaces

### Inner Products

#### Examples:
1. **Standard Inner Product**: In $\mathbb{R}^3$
   - $\langle (1,2,3), (4,5,6) \rangle = 1(4) + 2(5) + 3(6) = 32$

2. **Weighted Inner Product**: With weight matrix $W$

$$
W = \begin{pmatrix} 2 & 0 \\
0 & 1 \end{pmatrix}
$$
   
   - $\langle (1,1), (2,2) \rangle_W = (1,1)W(2,2)^T = 6$

4. **Function Inner Product**: On $[-1,1]$
   - $\langle f,g \rangle = \int_{-1}^1 f(x)g(x)dx$
   - For $f(x)=x$, $g(x)=x^2$: $\langle f,g \rangle = \int_{-1}^1 x^3dx = 0$
- **Definition**: $\langle u,v \rangle$ satisfying axioms
- **Norm**: $\|v\| = \sqrt{\langle v,v \rangle}$
- Example: Complex inner product:
  $\langle u,v \rangle = \sum_{i=1}^n u_i\overline{v_i}$

### Orthogonality

#### Examples:
1. **Orthogonal Vectors**: In $\mathbb{R}^2$
   - $v_1 = (3,4)$, $v_2 = (-4,3)$
   - Check: $\langle v_1,v_2 \rangle = 3(-4) + 4(3) = 0$

2. **Gram-Schmidt Process**: Starting with $\{(1,1,0), (1,0,1)\}$
   - $u_1 = (1,1,0)$
   - $u_2 = (1,0,1) - \frac{\langle (1,0,1),(1,1,0) \rangle}{\|(1,1,0)\|^2}(1,1,0)$
   - Results in orthogonal vectors

3. **Orthogonal Complement**: For subspace $V = \text{span}\{(1,1,0)\}$
   - $V^\perp = \{(x,y,z) | x + y = 0\}$
   - Example vector in $V^\perp$: $(1,-1,2)$
- **Orthogonal Complement**: $V^\perp = \{w | \langle w,v \rangle = 0 \text{ for all } v \in V\}$
- **Gram-Schmidt Process**:
  $v_k' = v_k - \sum_{i=1}^{k-1} \frac{\langle v_k,v_i\rangle}{\|v_i\|^2}v_i$
- Example: Orthogonalizing $\{(1,1,0), (1,0,1), (0,1,1)\}$
- 


## Eigendecomposition
- **Definition**: $A = PDP^{-1}$, where $D$ is diagonal and contains eigenvalues, and $P$ contains eigenvectors as columns.
- **Characteristic Polynomial**: $\det(A - \lambda I) = 0$

**Example**:

$$
A = \begin{bmatrix} 4 & 1 \\
2 & 3 \end{bmatrix}
$$

has eigenvalues $\lambda = 5, 2$.



## Jordan Forms
- **Definition**: $J$ is a block diagonal matrix of the form:

$$
J = \begin{bmatrix} \lambda & 1 & 0 \\
0 & \lambda & 1 \\
0 & 0 & \lambda \end{bmatrix}
$$

for each eigenvalue $\lambda$.

**Example**:

$$
A = \begin{bmatrix} 5 & 4 \\
0 & 5 \end{bmatrix}
$$

has Jordan form:

$$
J = \begin{bmatrix} 5 & 1 \\
0 & 5 \end{bmatrix}
$$



## Singular Value Decomposition (SVD)
- **Definition**: $A = U \Sigma V^T$, where:
  - $U, V$ are orthogonal.
  - $\Sigma$ is diagonal with singular values.

**Example**:

$$
A = \begin{bmatrix} 3 & 1 \\
1 & 3 \end{bmatrix}
$$

can be decomposed into $U$, $\Sigma$, and $V^T$.



## Matrix Factorizations
- **QR Decomposition**: $A = QR$ (orthogonal $Q$, upper triangular $R$).
- **LU Decomposition**: $A = LU$ (lower triangular $L$, upper triangular $U$).
- **Cholesky Decomposition**: $A = LL^T$ (for positive definite $A$).

**Example**:
For

$$
A = \begin{bmatrix} 4 & 12 \\
-5 & -13 \end{bmatrix},
$$

QR decomposition gives:

$$
Q = \begin{bmatrix} 0.8 & 0.6 \\
-0.6 & 0.8 \end{bmatrix}, \quad R = \begin{bmatrix} 5 & 15 \\
0 & 1 \end{bmatrix}.
$$



## Gradient, Divergence, Curl
- **Gradient**:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y} \\
\frac{\partial f}{\partial z} \end{bmatrix}
$$

- **Divergence**:

$$
\nabla \cdot \vec{F} = \frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} + \frac{\partial F_3}{\partial z}
$$

- **Curl**:

$$
\nabla \times \vec{F}
$$

**Example**:

$$
\vec{F} = (xy, yz, zx)
$$

Gradient:

$$
\nabla f = (y, z, x).
$$



## Jacobian Matrices and Determinants
- **Jacobian Matrix**:

$$
J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}
$$

- **Jacobian Determinant**:

$$
\det(J)
$$

measures local scaling.

**Example**:

$$
f(x, y) = (x^2, xy)
$$

$$
J = \begin{bmatrix} 2x & 0 \\
y & x \end{bmatrix}
$$



## Hessian Matrices
- **Definition**:

$$
H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}
$$

**Example**:

$$
f(x, y) = x^2 + y^2
$$

$$
H = \begin{bmatrix} 2 & 0 \\
0 & 2 \end{bmatrix}
$$



## Multiple Integrals and Change of Variables
- **Definition**:

$$
\int \int_R f(x, y) \, dx \, dy
$$

- **Change of Variables**: Substitute $x = g(u, v), y = h(u, v)$ with Jacobian determinant.

**Example**:
Switching to polar coordinates:

$$
  x = r\cos\theta, \quad y = r\sin\theta
$$



## Line and Surface Integrals
- **Line Integral**:

$$
\int_C \vec{F} \cdot d\vec{r}
$$

- **Surface Integral**:

$$
\int \int_S \vec{F} \cdot \vec{n} \, dS
$$

**Example**:

$$
\vec{F} = (y, -x), \quad C: x^2 + y^2 = 1
$$



## Theorems (Green, Stokes, Gauss)
- **Green's Theorem**: Relates a line integral to a double integral over the region $R$ bounded by $C$.

$$
\int_C \vec{F} \cdot d\vec{r} = \int \int_R \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dA
$$

- **Stokes' Theorem**: Generalizes Green’s theorem to 3D surfaces.
- **Gauss' Theorem**: Relates the flux through a surface to the divergence within the volume.



## Connections Between Topics
- **Linear Approximations**: Gradient and Jacobian relate to tangent planes.
- **Optimization Problems**:
  - Hessian for convexity.
  - Lagrange multipliers for constrained optimization.
- **Principal Component Analysis**: Uses eigendecomposition of the covariance matrix.
- **Coordinate Transformations**: Use Jacobians for multivariable transformations.
- **Differential Forms**: Generalize integrals to arbitrary dimensions.


