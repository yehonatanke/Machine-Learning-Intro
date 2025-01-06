# Linear Regression Summary

### Normal Equations in Linear Regression
The **Normal Equations** provide a direct formula to compute the optimal weights $\theta$ for minimizing the least squares error in linear regression. 
The formula is: $\theta = (X^T X)^{-1} X^T y$

Here:
- $X$: Feature matrix.
- $y$: Target values.
- $\theta$: Vector of optimal parameters.
- $X^T$: Transpose of the feature matrix.

The method computes $\theta$ without requiring iterative optimization, but matrix inversion makes it computationally expensive for large datasets.

---

### Comparison of Normal Equations and Gradient Descent

- **Normal Equations**:
  - Directly solve for $\theta$ using a closed-form solution.
  - Computationally expensive for large datasets due to matrix inversion.

- **Gradient Descent**:
  - Iteratively update $\theta$ by minimizing the cost function $J(\theta)$:<br>
  $J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x_i) - y_i \right)^2$
    
  - Updates weights using:<br>
    $\theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$
    
    where $\alpha$ is the learning rate.

Gradient descent is more scalable and suitable for large datasets.

---

### Objective of Linear Regression
In linear regression, the goal is not to separate data (as in classification) but to fit a line (or hyperplane) that best models the relationship between features $X$ and the target $y$. This is achieved by minimizing the **sum of squared errors**.

For example:
- In simple linear regression (1 feature):<br>
  $y = \theta_0 + \theta_1 x$
- In multiple linear regression (n features):<br>
  $y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$

---

### Adjusting Weights to Fit the Data
The process of fitting the model involves:
1. **Initializing Weights**: Start with random or zero values for $\theta$.
2. **Making Predictions**: Compute predictions using the current weights:<br>
  $h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$
3. **Computing Error**: Calculate the error using a cost function such as the Mean Squared Error.
4. **Updating Weights**: Optimize $\theta$ iteratively (e.g., using gradient descent).
5. **Convergence**: Repeat until the cost function is minimized, finding the "best" weights.

---

### Summary Statement
Linear regression involves adjusting weights and biases to minimize prediction error, aiming to model a continuous relationship between input features and output targets. Both the **Normal Equations** and **Gradient Descent** methods achieve this goal, but their suitability depends on the dataset size and computational efficiency requirements.
