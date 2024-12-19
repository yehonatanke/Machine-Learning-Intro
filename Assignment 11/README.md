# Assignment 11

## Task Overview
- **Topics Covered**: Perceptron, Linear Regression, Logistic Regression

## Objective: Handwritten Digit Classification
The assignment focuses on the implementation, training, and evaluation of machine learning models for digit recognition using the MNIST dataset.

## Implementation Guidelines
- **Language**: Python.
- **Models**: Implement Perceptron, Logistic Regression, and Linear Regression from scratch.
- **Performance**: Ensure reasonable runtime and efficiency with vectorized operations using NumPy.
- **Documentation**: Write clear, well-commented code.
- **Deliverables**:
  1. A Jupyter notebook.
  2. Necessary executable files.
  3. A written report summarizing results.

---

## Introduction to MNIST Dataset
The MNIST dataset consists of grayscale 28x28 pixel images of handwritten digits (0-9). 
- **Key Features**:
  - **Size**: 28x28 pixels (784 features per image).
  - **Grayscale**: Pixel values range from 0 to 255.
  - **Label**: A single digit (0-9).
- **Data Split**:
  - Training Set: 60,000 images.
  - Test Set: 10,000 images.

---

## Part A: Perceptron Learning Algorithm
1. **Extension to Multi-Class**:
   - Use the **One-vs-All Strategy**.
   - Train 10 binary classifiers.
   - Output the label with the maximum confidence score.

2. **Algorithm**:
   - Initialize weights.
   - For each misclassified example, update weights using:
     $$w(t+1) = w(t) + y(t)x(t)$$
   - Continue until no misclassified examples remain.

3. **Pocket Algorithm**:
   - Stores the best-performing weight vector during updates to enhance stability.

### Tasks
- **A1**: Apply multi-class perceptron to MNIST.
- **A2**: Compute confusion matrix and accuracy (ACC).
- **A3**: Generate confusion tables and sensitivity (TPR) for each digit.
- **A4**: Discuss results.

---

## Part B: Softmax Regression
1. **Objective**:
   - Extend logistic regression to classify multiple classes (0-9).
   - Estimate probabilities $P(y = k|x)$.

2. **Cost Function**:
   <br>Minimize:<br>
   $$E_{in}(w) = -\sum_{n=1}^N \sum_{k=1}^K 1\{y_n = k\} \log \frac{e^{w_k^T x_n}}{\sum_{j=1}^K e^{w_j^T x_n}}$$

### Tasks
- **B1**: Implement gradient descent for the softmax cost function.
- **B2**: Evaluate confusion matrix and accuracy.
- **B3**: Compute sensitivity (TPR) for each class.
- **B4**: Discuss results.

---

## Part C: Linear Regression
1. **Objective**:
   - Treat digit classification as a regression task.
   - Use least squares method.

### Tasks
- **C1**: Formulate classification as linear regression.
- **C2**: Evaluate performance on the test set.
- **C3**: Compare results with perceptron and softmax regression.
- **C4**: Discuss limitations of linear regression for classification.

---

**Appendix**<br>
Fetching the MNIST Dataset
Use `fetch_openml` from `scikit-learn`:
```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
```
