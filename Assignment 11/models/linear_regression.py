
"""
## PART C: Linear Regression

### **Thought Process:**

**Formulating the Problem as a Linear Regression Task:**<br>
The task at hand is the classification of handwritten digits, which traditionally represents a classification problem. However, linear regression is generally used for predicting continuous variables, not categorical ones. To adapt linear regression to this context, we frame it as a multi-output regression problem, where the objective is to predict the digit labels as numerical values.

A significant challenge arises because the output values are discrete integers (ranging from 0 to 9), differing from the continuous outputs in typical regression settings.

To address this, we:
  - Apply one-hot encoding to the target labels, transforming each digit into a binary vector where the position corresponding to the correct digit is set to 1 (e.g., the digit 1 becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).
  - Conceptualize the problem as "multi-class regression," where the model predicts the one-hot encoded vector for each digit.
  - Represent each image as a feature vector consisting of 784 pixel intensities (one for each pixel).
  
**Evaluating Linear Regression on the Test Set:**<br>
  - Performance is assessed by how effectively the model predicts digit labels on the test set.
  - Since linear regression outputs continuous values, a thresholding approach is used to map the predictions back to discrete classes.
  - Evaluation metrics may include accuracy (as a classification metric) along with potential metrics such as mean squared error or cross-entropy loss (more common in regression tasks).
  - Given that linear regression is not ideal for categorical tasks, especially with the complexity and high dimensionality of handwritten digits, we expect its performance to be subpar.

### **Plan for Implementation:**

**Data Preparation:**
  - Load the MNIST dataset.
  - Flatten the 28x28 pixel images into 784-dimensional vectors.
  - One-hot encode the digit labels (0-9).

**Training the Linear Regression Model:**
  - Utilize the least squares method to train the linear regression model on the flattened image data.
  - The weight matrix is computed using the formula:
  $$
  W = (X^T X)^{-1} X^T y
  $$
  where $X$ represents the matrix of input images size  $N \times 784$, and $y$ is the matrix of one-hot encoded labels.

**Performance Evaluation:**<br>
  - Use the trained model to predict digit labels on the test set.
  - Convert the continuous predicted values to discrete labels by selecting the class with the highest predicted value.
  - Compute classification accuracy by comparing the predicted labels with the actual labels.

### **Linear Regression Approach:**

- **Advantages:**
   - Simplicity in both implementation and concept.
   - Fast training on small datasets.
  
- **Disadvantages:**
   - Linear regression is fundamentally unsuitable for classification tasks.
   - It presupposes a linear relationship between the features and the target, which is unrealistic for image data such as handwritten digits.
   - The model may fail to generalize effectively, leading to suboptimal accuracy when compared to more appropriate classification algorithms like logistic regression, support vector machines, or neural networks.

### **MulticlassLinearRegression - Model Overview**

- `self.weights`: Stores the learned weights after training.
- `self.training_errors`: Tracks errors during training (not actively used in the code).
- `self.test_errors`: Tracks errors during testing (not actively used in the code).

**Fit Method (`fit`)**:
- **Purpose**: This method trains the model using the normal equation for linear regression.
- **Formula**: The weights are computed using the formula:
  $$
  w = (X^T X)^{-1} X^T y
  $$
  Where $X$ is the feature matrix (size: $n_{\text{samples}} \times n_{\text{features}}$) and $y$ is the one-hot encoded label matrix (size: $n_{\text{samples}} \times n_{\text{classes}}$).
- **Procedure**:
  - It calculates the pseudo-inverse of $X^T X$, multiplies it by $X^T$, and then multiplies by the one-hot encoded labels $y$ to obtain the weight matrix `self.weights`.
   
3. **Predict Method (`predict`)**:
   - **Purpose**: This method makes predictions for the input data based on the learned weights.
   - **Procedure**:
     - Computes the scores by multiplying the feature matrix $X$ by the weight matrix $w$.
     - Uses `np.argmax` to select the class with the highest score for each sample, thus predicting the class label.

4. **Predict Proba Method (`predict_proba`)**:
   - **Purpose**: This method returns the raw regression scores (before applying `argmax`), which can be useful for evaluating errors or understanding the confidence of predictions.
   - **Procedure**:
     - Computes the scores by multiplying the feature matrix $X$ by the weight matrix $w$, without applying any thresholding or argmax.
   
### **Evaluation Function (`evaluate_model`)**:

This function calculates and prints the classification accuracy of the model. It takes the true labels (`y_true`) and the predicted labels (`y_pred`) as inputs, calculates the proportion of correct predictions, and prints the accuracy.

### **Accuracy Score Function (`accuracy_score`)**:

This function calculates the classification accuracy as the ratio of correct predictions to the total number of samples. It returns the accuracy as a float.


### **Potential Improvements or Notes**:
- **Error Tracking**: The `training_errors` and `test_errors` attributes are initialized but not actively used. Implementing error tracking could provide insight into how well the model generalizes.
- **Regularization**: The model uses the normal equation for fitting, which may lead to overfitting in cases with high-dimensional data (like MNIST). Regularization techniques (e.g., L2 regularization) might improve performance.
- **Model Comparison**: While the `plot_comparison` function is defined, it's not invoked in the provided code. Adding functionality to compare multiple models might be useful for evaluating different algorithms.

This model is a basic implementation of a multiclass linear regression algorithm, suitable for understanding how linear regression can be applied to classification tasks, though it may struggle with complex datasets due to the limitations of linear regression for classification.
"""

class MulticlassLinearRegression:
    def __init__(self):
        self.weights = None
        self.training_errors = []
        self.test_errors = []

    def fit(self, X, y):
        """
        Fit linear regression using normal equation: w = (X^T X)^(-1) X^T y
        Args:
            X: Training features (n_samples, n_features)
            y: One-hot encoded labels (n_samples, n_classes)
        """
        # Calculate weights using normal equation
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X
        Returns class with highest regression value
        """
        scores = X @ self.weights
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        """
        Get raw regression values (before argmax)
        Useful for calculating errors
        """
        return X @ self.weights

def evaluate_model(y_true, y_pred, model_name="Linear Regression"):
    """Calculate and print model performance metrics"""
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples

    print(f"\n{model_name} Results:")
    print(f"Classification Accuracy: {accuracy:.4f}")
    return accuracy

def plot_comparison(accuracies, model_names):
    """Plot bar chart comparing model accuracies"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title("Model Comparison: Classification Accuracy", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def accuracy_score(y_true, y_pred):
    """
    Calculate the classification accuracy as the proportion of correct predictions.

    Args:
        y_true: True class labels (n_samples,)
        y_pred: Predicted class labels (n_samples,)

    Returns:
        accuracy: Classification accuracy as a float
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

# Load data
X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = prepare_data()

# Train linear regression model
print("Training linear regression model...")
linear_regression_model = MulticlassLinearRegression()
linear_regression_model.fit(X_train, y_train)

# Make predictions
lr_y_pred = linear_regression_model.predict(X_test)

# Evaluate and visualize results
lr_accuracy = evaluate_model(y_test_orig, lr_y_pred)

# Plot confusion matrix
model_details = {
    'model_name': 'Linear Regression',
}
plot_confusion_matrix(y_test_orig, lr_y_pred, model_details)
print_classifier_analysis(y_test_orig, lr_y_pred, model_name="Linear Regression")

"""####  **Linear Regression Results**
The results of the linear regression model on the MNIST dataset reveal a mixed performance, with notable strengths and weaknesses across different digit classes. The model performs well on simpler, more distinct digits such as zero, one, and two, demonstrating high precision and recall, indicating effective classification. Specifically, digit one shows particularly high recall, although its precision is slightly lower, suggesting that the model identifies this digit well but occasionally misclassifies it as another. However, the model struggles with more complex digits like five, eight, and nine, which exhibit lower recall and F1-scores, especially for digit five, where recall is notably reduced. These difficulties are likely due to the inherent visual similarities between these digits and others, which linear regression, with its limited ability to model non-linear decision boundaries, cannot distinguish effectively. The overall classification accuracy reflects the model’s general competence but also highlights its limitations in handling the subtle and overlapping features of handwritten digits. Consequently, while linear regression may serves as a solid baseline, its performance on more complex digit classes underscores the necessity for more advanced models to achieve higher classification accuracy and better generalization.

####  **Comparison of Linear Regression, Perceptron, and Softmax Regression for MNIST Classification**
When comparing linear regression, the perceptron algorithm, and softmax regression for the task of MNIST digit classification, it becomes clear that each model has unique strengths and weaknesses, particularly in their ability to handle complex data patterns inherent in the dataset.

**Linear Regression**:
Linear regression is the simplest of the three models and is based on the assumption that there is a linear relationship between the input features (pixel values) and the output (digit class). While it can be trained quickly and easily, its simplicity is both a strength and a significant limitation. The major weakness of linear regression lies in its inability to capture non-linear patterns in the data. Handwritten digits exhibit complex variations in shape, size, and orientation, which cannot be effectively modeled with a linear decision boundary. This results in poor classification performance, particularly for digits with similar visual features. Additionally, the squared error loss function used in linear regression does not effectively penalize misclassifications in classification tasks, leading to suboptimal decisions. Therefore, while linear regression can serve as a baseline model, it is ill-suited for MNIST's complexity.

**Perceptron Algorithm**:
The perceptron algorithm, in contrast, uses a linear decision boundary to separate each digit class in a one-vs-all (OvA) fashion. While this allows the model to extend linear regression to multi-class classification, the perceptron still operates within the confines of linear decision boundaries. Its strength lies in its ability to adjust weights incrementally, making it capable of learning from the data over time. However, its reliance on linear boundaries limits its capacity to effectively separate complex data patterns, especially when dealing with similar digits, such as 5, 8, and 9. This results in reduced accuracy for certain classes, even though the perceptron can achieve decent overall accuracy (typically above 80%). The perceptron model also struggles with noisy data and overlapping classes, as its learning rule is based on simple binary updates rather than a probabilistic approach. Despite these limitations, the perceptron offers a significant improvement over linear regression due to its ability to handle multi-class classification and incremental learning.

**Softmax Regression**:
Softmax regression, also known as multinomial logistic regression, improves upon both linear regression and the perceptron by modeling the probabilities of each class rather than relying on a binary decision boundary for each class. By using the softmax function, which transforms the raw output scores into class probabilities, softmax regression offers a more nuanced understanding of the data. This model performs significantly better on MNIST than linear regression or the perceptron, achieving higher classification accuracy (typically above 90%). The softmax function allows the model to assign higher confidence to the correct class and lower confidence to incorrect ones, which aids in distinguishing between visually similar digits. However, like the perceptron, softmax regression still operates with linear decision boundaries, meaning it cannot fully capture non-linear relationships. Despite this, softmax regression’s ability to model probabilities and output more meaningful decision scores gives it a clear advantage over both linear regression and the perceptron.

**Conclusion:**
In terms of overall effectiveness for the MNIST classification task, **softmax regression** is the best model among the three, as it strikes a balance between simplicity and performance. While it still uses linear decision boundaries, its probabilistic approach allows for more nuanced predictions and higher accuracy, especially when compared to linear regression and the perceptron. **The perceptron** provides a useful improvement over linear regression by handling multi-class classification, but its performance is limited by its linearity and struggles with overlapping classes. **Linear regression**, though fast and simple, is not suited for complex, non-linearly separable data and thus performs poorly on MNIST. However, all three models are outperformed by more advanced models, such as neural networks, which can learn non-linear patterns and achieve even higher levels of accuracy.
"""