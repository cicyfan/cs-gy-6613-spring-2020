---
title: Linear Algebra for Machine Learning
---

# Linear Algebra for Machine Learning

The corresponding chapter of Ian Goodfellow's Deep Learning is essentially the background you need.

<iframe src="https://www.deeplearningbook.org/contents/linear_algebra.html" width="800" height="1200"></iframe>

## Key Points 

We can now summarize the points to pay attention to for ML applications.  In the following we assume a data matrix $A$ with $m$ rows and $n$ columns. We also assume that the matrix is such that it has $r$ independent rows, called *the matrix rank*.

### The Four Fundamental Subspaces

![Four fundamental spaces](images/four-fundamental-spaces-linear-alg.png)

The *fundamental theorem of Linear Algebra* specifies the effect of the multiplication operation of the matrix and a vector ($A\mathbf{x}$). The matrix gives raise to 4 subspaces:

1. **The column space of $A$**, denoted by $\mathcal{R}(A)$, with dimension $r$.
2. **The nullspace of $A$**, denoted by $\mathcal{N}(A)$, with dimension $n-r$.
3. **The row space of $A$** which is the column space of $A^T$, with dimension $r$
4. **The left nullspace of $A$**, which is the nullspace of $A^T$, denoted by $\mathcal{N}(A^T)$, with dimension $m-r$. 

The real action that the matrix perform is to **transform** its row space to its column space. 

The type of matrices that are common in ML are those that the number of rows $m$ representing observations is much larger than the number of columns $n$ that represent features. We will call these matrices "tall" for obvious reasons. Let us consider one trivial but instructive example of the smallest possible "tall" matrix:

$$\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21}  & a_{22} \\\\ a_{31} & a_{32} \end{bmatrix} = \begin{bmatrix} 1       & 0 \\\\ 5       & 4 \\\\ 2       & 4 \end{bmatrix}$$

In ML we are usually concerned with the problem of learning the weights $x_1, x_2$ that will combine the features and result into the given target variables $\mathbf{b}$. The notation here is different and we have adopted the notation of many linear algebra textbooks. 

$$ \begin{bmatrix} 1       & 0 \\\\ 5       & 4 \\\\ 2       & 4 \end{bmatrix}  \begin{bmatrix} x_1 \\\\ x_2  \end{bmatrix} =
\begin{bmatrix} b_1 \\\\ b_2 \\\\  b_3  \end{bmatrix}$$

To make more explicit the combination of features we can write,

$$ x_1 \begin{bmatrix} 1 \\\\ 5 \\\\ 2 \end{bmatrix} + x_2 \begin{bmatrix} 0 \\\\ 4 \\\\  4  \end{bmatrix} = \begin{bmatrix} b_1 \\\\ b_2 \\\\  b_3  \end{bmatrix}$$

Since $m=3 > n=2$, we have more equations than unknowns we in general we have no solutions - a system with $m > n$ will be solvable only for certain right hand sides $\mathbf{b}$. Those are all the vectors $\mathbf{b}$ that lie in the column space of $A$.

![column-space](images/column-space.png)

In this example, as shown in the picture $\mathbf{b}$ must lie in the plane spanned by the two columns of $A$. The plane is a subspace of $\mathbb{R}^m=\mathbb{R}^3$ in this case. 

Now instead of looking at what properties $\mathbf{b}$ must have for the system to have a solution, lets look at the *dual* problem i.e. what weights $\mathbf{x}$ can attain those $\mathbf{b}$. The right-hand side $\mathbf{b}=0$ always allows the solution $\mathbf{x}=0$ 
The solutions to $A \mathbf{x} = \mathbf{0}$ form a vector space - **the nullspace** $\mathcal{N}(A)$. The nullspace is also called the *kernel* of matrix $A$ and the its dimension $n-r$ is called the nullity. 

$\mathcal{N}(A)$ is a subspace of $\mathbb{R}^n=\mathbb{R}^2$ in this case. For our specific example,

$$ x_1 \begin{bmatrix} 1 \\\\ 5 \\\\ 2 \end{bmatrix} + x_2 \begin{bmatrix} 0 \\\\ 4 \\\\  4  \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\  0  \end{bmatrix}$$

the only solution that can satisfy this set of homogenous equations is: $\mathbf{x}=\mathbf{0}$ and this means that the null space contains only the zero vector and this 

Two vectors are independent when their linear combination cannot be zero, unless both $x_1$ and $x_2$ are zero.  The columns of $A$ are therefore linearly independent and they span the column space. They have therefore all the properties needed for them to constitute a set called the *basis* for that space and we have two basis vectors (the rank is $r=2$ in this case). The dimension of the column space is in fact the same as the dimension of the row space ($r$) and the mapping from row space to column space is in fact invertible. Every vector $\mathbf{b}$ comes from one and only one vector $\mathbf{x}$ of the row space ($\mathbf{x}_r$). And this vector can be found by the inverse operation - noting that only the inverse $A^{-1}$ is the operation that moves the vector correctly from the column space to the row space. The inverse exists only if $r=m=n$ - this is important as in most ML problems we are dealing with "tall" matrices with the number of equations much larger than the number of unknowns which makes the system *inconsistent* (or *degenerate*).

![projection-column-space](images/projection-column-space.png)
*Projection onto the column space*

Geometrically you can think about the basis vectors as the axes of the space. However, if the axes are not orthogonal, calculations will tend to be complicated not to mention that we usually attribute to each vector of the basis to have length one (1.0). 

### Eigenvalues and Eigenvectors
The following video gives an intuitive explanation of eigenvalues and eigenvectors and its included here due to its visualizations that it offers.  The video must be viewed in conjunction with [Strang's introduction](http://math.mit.edu/~gs/linearalgebra/linearalgebra5_6-1.pdf)

<iframe width="560" height="315" src="https://www.youtube.com/embed/PFDu9oVAE-g" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

During the lecture we will go through an example from how your brain processes the sensory input generated by the voice of the lecturer(unless you are already asleep by that time) to combine *optimally* the sound from both your ears.

A geometric interpretation of the eigenvectors and eigenvalues is given in the following figure:

![eigenvectors](images/eigenvectors.png)

