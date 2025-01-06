# Perceptron Learning Algorithm

The **Perceptron Learning Algorithm (PLA)** is a foundational method in the field of machine learning and artificial intelligence, used to train a linear classifier. Introduced by Frank Rosenblatt in 1958, it is a supervised learning algorithm designed for binary classification tasks, wherein the objective is to find a linear decision boundary that separates data points into two distinct classes.

## Assumptions and Prerequisites

1. **Linearly Separable Data**: The algorithm assumes that the dataset is linearly separable, meaning there exists a hyperplane that can perfectly divide the classes.
2. **Feature Space**: Input data is represented as feature vectors, and each vector is associated with a corresponding label (+1 or -1).
3. **Learning Rule**: The algorithm employs a weight vector $w$ and a bias term $b$ to define the linear decision boundary.

## Algorithm Description

The algorithm iteratively adjusts the weights and bias using the following steps:

1. **Initialization**: 
   - Start with an initial weight vector $w$ and bias $b$, often initialized to zero or small random values.
   - Define the learning rate $\eta$, a positive constant that controls the magnitude of updates.

2. **Iteration**:
   - For each data point $(x_i, y_i)$ in the training set:
     - Compute the prediction: $\hat{y} = \text{sign}(w \cdot x_i + b)$.
     - If $\hat{y} \neq y_i$ (misclassification):
       - Update the weights: $w \leftarrow w + \eta \cdot y_i \cdot x_i$.
       - Update the bias: $b \leftarrow b + \eta \cdot y_i$.

3. **Convergence**:
   - Repeat the iteration until all points are correctly classified (if data is linearly separable) or a predefined maximum number of iterations is reached.

## Theoretical Insights

- **Convergence Guarantee**: For linearly separable data, the PLA is guaranteed to converge to a solution within a finite number of iterations.
- **Limitations**:
  - If the data is not linearly separable, the algorithm fails to converge, cycling indefinitely.
  - The solution is not unique; the resulting hyperplane depends on the initialization and order of data points.
  
## Variants and Extensions

- **Relaxation Techniques**: Allowing classification errors for non-linearly separable data.
- **Kernelized Perceptron**: Incorporating kernel functions to handle non-linear decision boundaries.
- **Multi-class Perceptron**: Adapting the algorithm for multi-class classification problems.

## Python Implementation (Example)

```python
class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        """
        Train the Perceptron on the given dataset.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), the feature matrix.
        - y: numpy array of shape (n_samples,), the labels (+1 or -1).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                
                # Update weights and bias if misclassified
                if y_predicted != y[idx]:
                    self.weights += self.learning_rate * y[idx] * x_i
                    self.bias += self.learning_rate * y[idx]

    def predict(self, X):
        """
        Predict class labels for the given input data.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), the feature matrix.
        
        Returns:
        - numpy array of predicted labels (+1 or -1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

def run_example():
    # Training data: 4 points in a 2D feature space
    X = np.array([[2, 3], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, -1, -1])  # Labels: +1 and -1

    # Initialize and train the perceptron
    perceptron = Perceptron(learning_rate=0.1)
    perceptron.fit(X, y)

    # Predict on new data
    X_test = np.array([[2, 2], [3, 3]])
    predictions = perceptron.predict(X_test)

    print("Predictions:", predictions)
```
