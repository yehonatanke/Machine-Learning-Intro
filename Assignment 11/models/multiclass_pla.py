
"""## **Part A: Perceptron Learning Algorithm**

### **Multiclass PLA V1 - Model Overview**

**Multi-class Perceptron and One-vs-All Strategy**<br>
The perceptron algorithm, initially designed for binary classification, is extended to multi-class problems using a **one-vs-all strategy**:  
- For each class $k$, the perceptron distinguishes between samples of that class (positive) and all others (negative).  
- This creates $C$ binary classifiers (where $C$ is the number of classes).  
- During prediction, the classifier assigns a sample to the class with the highest score (dot product of input and weights).  

**Update Rule**<br>
The perceptron updates its weights when a sample is misclassified:  
$$
w_{k} = w_{k} + y \cdot x
$$
Where:  
- $w_{k}$: Weights for class $k$.  
- $y$: Label $+1$ for positive class, $-1$ otherwise.  
- $x$: Feature vector of the misclassified sample.

**Pocket Algorithm**<br>
To handle non-linearly separable cases:  
- The **Pocket Algorithm** retains the best-performing weights (based on error) during training, ensuring stability even when convergence isn't guaranteed.


### **Model Implementation**

#### Model Structure and Mechanisms
The `MulticlassPerceptron_V1` class is designed for flexibility, modularity, and performance. Here's an explanation of its key elements:

1. **Attributes**:  
   - `weights` and `pocket_weights` track current and best weights for each class.  
   - `training_errors` and `test_errors` log error rates during training for analysis.  

2. **Initialization Choices**:  
   - Weight initialization starts at zero for simplicity and reproducibility.  
   - The `max_iter` parameter controls the maximum training epochs, balancing training time and performance.  

3. **Vectorization**:  
   - Operations like dot products and error calculations are vectorized for efficiency, especially given the high dimensionality of MNIST data.  

4. **Binary Label Conversion**:  
   - The `_get_binary_labels` method ensures each class's binary classification is seamlessly handled.  

5. **Error Calculation**:  
   - Error rates are computed as the proportion of misclassified samples, allowing easy comparison and pocket weight updates.  


#### **Code Summary**

The `MulticlassPerceptron_V1` implementation follows these key steps:  

1. **Weight Initialization**:  
   - Both current and pocket weights are initialized, ensuring a baseline for comparison.  

2. **Training Loop**:  
   - Each epoch processes all classes, updating weights and errors as necessary.  
   - Pocket weights are updated only when a better solution is found.  

3. **Prediction**:  
   - During inference, the class with the highest score is selected, ensuring consistency with the multi-class approach.  

4. **Sensitivity Calculation**:  
   - Evaluates performance per class, aiding in understanding the model's strengths and weaknesses.


#### **Reasoning Behind Design Choices**

- **Pocket Algorithm**:
- **Error Logging**: Provides insights into model behavior during training and testing phases.  
- **Progress Indicators**: Nested progress bars offer detailed feedback.
- **Vectorized Operations**:
---

#### **Approach Summary**

The implementation reflects a balance between simplicity (e.g., zero initialization) and advanced features (e.g., Pocket Algorithm). The model prioritizes interpretability, efficiency, and adaptability to real-world scenarios, aligning with modern best practices in machine learning.
"""

class MulticlassPerceptron_V1:
    """
    A multiclass perceptron implementation using the Pocket Algorithm.

    Attributes:
        n_classes (int): Number of target classes. Default is 10.
        max_iter (int): Maximum number of iterations for training. Default is 1000.
        weights (numpy.ndarray): Current weights for each class.
        pocket_weights (numpy.ndarray): Best weights (least error) retained for each class.
        best_errors (numpy.ndarray): Minimum errors recorded for each class.
        training_errors (list): List to track training error per epoch.
        test_errors (list): List to track test error per epoch.
        pocket_weights_history (list): List to track pocket weights per epoch.
    """
    def __init__(self, n_classes=10, max_iter=1000):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.weights = None  # To store current weights for each class
        self.pocket_weights = None  # To store best weights based on error
        self.best_errors = None  # To track the lowest errors for each class
        self.training_errors = []  # To log training error for each epoch
        self.test_errors = []  # To log test error for each epoch
        self.pocket_weights_history = []  # To log pocket weights for each epoch

    def _init_weights(self, n_features):
        """
        Initializes the weight matrices and pocket weights.

        Args:
            n_features (int): Number of features in the input data.
        """
        # Initialize weights to zero for all classes and features
        # Creates a matrix of shape (number of classes, number of features)
        self.weights = np.zeros((self.n_classes, n_features))

        # Pocket weights start as a copy of the initial weights
        self.pocket_weights = self.weights.copy()

        # Initial best errors set to 1.0 (worst case scenario - all misclassified)
        # This is an array with one element per class
        self.best_errors = np.ones(self.n_classes)

    def _get_binary_labels(self, y):
        """
        Converts one-hot encoded labels into binary labels for all classes simultaneously.

        Args:
            y (numpy.ndarray): One-hot encoded labels.

        Returns:
            numpy.ndarray: Binary labels of shape (n_samples, n_classes).
        """
        return 2 * y - 1  # Convert 1 to +1 and 0 to -1 for all classes

    def _calculate_errors(self, X, y, weights):
        """
        Calculates the classification errors for all classes simultaneously.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): One-hot encoded labels.
            weights (numpy.ndarray): Weights used for prediction.

        Returns:
            numpy.ndarray: Fraction of misclassified samples per class.
        """
        binary_y = self._get_binary_labels(y) # Binary labels for current class
        scores = X @ weights.T # Compute scores as dot product of X and weights
        predictions = np.where(scores > 0, 1, -1) # Predict class based on sign of scores
        return np.mean(predictions != binary_y, axis=0) # Calculate proportion of predictions

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Trains the perceptron model using the Pocket Algorithm.

        Args:
            X_train (numpy.ndarray): Training data matrix of shape (n_samples, n_features).
            y_train (numpy.ndarray): One-hot encoded training labels.
            X_test (numpy.ndarray): Test data matrix.
            y_test (numpy.ndarray): One-hot encoded test labels.
        """
        n_samples, n_features = X_train.shape # Get number of samples and features
        self._init_weights(n_features) # Initialize weights

        progress_bar = tqdm(range(self.max_iter), desc="Training Progress", unit="epoch", leave=True, position=0)

        for iteration in progress_bar:
            # Compute scores and determine misclassified samples for all classes
            binary_y_train = self._get_binary_labels(y_train) # Binary labels for the current class
            scores = X_train @ self.weights.T
            misclassified = (scores * binary_y_train <= 0)  # Shape: (n_samples, n_classes)

            # Nested progress bar for processing each class
            class_progress = tqdm(range(self.n_classes), desc="Processing Classes", unit="class", leave=False, position=1)

            # Update weights for misclassified samples
            for class_idx in class_progress:
                if np.any(misclassified[:, class_idx]):
                    update_idx = np.where(misclassified[:, class_idx])[0][0]
                    self.weights[class_idx] += binary_y_train[update_idx, class_idx] * X_train[update_idx]

            # Calculate errors for all classes
            current_errors = self._calculate_errors(X_train, y_train, self.weights)
            improved_mask = current_errors < self.best_errors

            # Update pocket weights where errors improved
            self.pocket_weights[improved_mask] = self.weights[improved_mask].copy()
            self.best_errors[improved_mask] = current_errors[improved_mask]

            # Calculate and log training and test errors
            avg_train_error = np.mean(self.best_errors)
            avg_test_error = np.mean(self._calculate_errors(X_test, y_test, self.pocket_weights))

            self.training_errors.append(avg_train_error)
            self.test_errors.append(avg_test_error)

            if iteration % 100 == 0:
                self.pocket_weights_history.append(self.pocket_weights.copy())

            progress_bar.set_postfix({'Train Error': f'{avg_train_error:.4f}', 'Test Error': f'{avg_test_error:.4f}'})

    def predict(self, X):
        """
        Predicts the class labels for input samples using the best weights.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        scores = X @ self.pocket_weights.T # Compute scores for all classes
        return np.argmax(scores, axis=1) # Select class with highest score

    def calculate_sensitivity(self, X, y_true):
        """
        Calculates sensitivity (TPR) for each class.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            y_true (numpy.ndarray): True class labels as a 1D array.

        Returns:
            numpy.ndarray: Sensitivity values for each class.
        """
        y_pred = self.predict(X)
        sensitivities = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            true_positives = np.sum((y_true == i) & (y_pred == i))
            false_negatives = np.sum((y_true == i) & (y_pred != i))
            sensitivities[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return sensitivities

"""#### **Model Expectations**
Before running the model let's examine our expectations of the model and what might happen
###### **Challenges with PLA (Perceptron Learning Algorithm)**
- **Multi-Class Complexity**:
  - PLA is designed for linear classification, and handling multi-class problems involves converting them into multiple binary classification tasks (e.g., using a one-vs-all strategy).
  - Each binary classifier learns to distinguish one class from all others, but this increases the number of decision boundaries and heightens the chances of conflicts between them.
  
- **Data Linearity**:
  - PLA assumes data is linearly separable. However, MNIST's digit classes are not perfectly linearly separable in the feature space, especially for similar digits (e.g., '3' vs. '8').
  - The lack of flexibility in PLA's decision boundaries likely means that it will fail to fully classify such data accurately.

###### **Expected Behavior During Training**
- **Convergence Likelihood**:
  - Due to the linear nature of PLA, it is unlikely to converge on MNIST. Non-linearly separable regions in the data will prevent the algorithm from finding a consistent weight vector for all samples.
  - The model may oscillate between solutions or settle on suboptimal weights, leading to higher error rates.

- **Error Trends**:
  - **Training Error**: Likely to decrease initially but plateau at a significant value due to the inability to separate classes perfectly.
  - **Test Error**: Expected to remain high, reflecting the model's limited generalization ability for MNIST.

---

Now, let's create and instance of the model and run it:
"""

def load_and_preprocess_data2(test_size=0.2):
    """
    Load and preprocess MNIST dataset:
    - Load data using fetch_openml
    - Normalize pixel values to [0,1]
    - Add bias term
    - Convert labels to one-hot encoding
    - Split into train/test sets
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    # Normalize features to [0,1]
    X = X / 255.0

    # Add bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Convert labels to integers
    y = y.astype(int)

    # Convert to one-hot encoding
    y_onehot = np.zeros((y.shape[0], 10))
    y_onehot[np.arange(y.shape[0]), y] = 1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42
    )

    # Also return original labels for confusion matrix
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, y_train_orig, y_test_orig

# Prepare the data again in an orderly manner to ensure correct loading of the data
X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = prepare_data()

# Set max_iter=500 as testing showed no significant improvement beyond this point.
max_iter = 500
pla_model_name = "Perceptron Learning Algorithm V1"

# Model details (for the plots)
pla_model_details = {
    'model_name': pla_model_name,
    'epochs': max_iter
}

# Create and train model
pla_model = MulticlassPerceptron_V1(max_iter=max_iter)

print("Training model...")
pla_model.fit(X_train, y_train, X_test, y_test)

# Make predictions
pla_y_pred = pla_model.predict(X_test)

"""#### **Model: Multiclass PLA V1 - Results**
For the following steps, we will be utilizing functions from the utility section to streamline our process.

**Classifier Analysis**<br>
We will now display the results of the classifier analysis to assess the performance metrics such as accuracy, precision, recall, and F1 score. This will provide us with a comprehensive overview of the classifier's effectiveness.
"""

# Print classifier analysis
print_classifier_analysis(y_test_orig, pla_y_pred, model_name=pla_model_name)

"""The performance metrics for the Perceptron Learning Algorithm (PLA) on the multi-class classification task reveal several key insights about the model's behavior and effectiveness:

**Precision, Recall, and F1-Score Distribution:**<br>
   - The precision values across the different digit classes show variation, with higher precision for digits 1, 0, and 7, while digits like 5, 3, and 4 have notably lower precision values. This suggests that the model is more accurate in predicting certain digits, particularly those with more distinct features, while struggling more with others, possibly due to class imbalance or similar features between certain digits.
   - Recall values follow a similar pattern, with high recall for digits 1, 0, and 6, but a significant drop in recall for digits 8 and 5. The model struggles more to identify certain digits, particularly 8, which likely indicates that there is a significant challenge in distinguishing these digits from others in the dataset.
   - The F1-scores, which balance precision and recall, confirm these trends. The F1-scores for digits 1 and 0 are the highest, while digits 5, 8, and 9 have relatively lower F1-scores, highlighting that the model performs well on some digits but faces difficulties with others.

**Overall Model Performance:**<br>
The overall classification accuracy suggests that the model is surprisingly effective for this task, though there is still room for improvement. The model’s ability to correctly classify most digits is decent, but it is clear that certain digits (like 8 and 5) lead to significant misclassifications.
   
**Analysis of Accuracy:**<br>
The accuracy per digit mirrors the patterns in precision and recall, with digits 1, 0, and 6 having high accuracy, while digits like 8 and 5 show lower accuracy. This indicates that, although the model can accurately classify certain digits, it struggles with others, likely due to the lack of flexibility in the PLA to capture more complex decision boundaries for those specific digits.

**Potential Causes and Areas for Improvement:**<br>
   - The Perceptron Learning Algorithm, while effective for simpler problems, might not be complex enough to capture the nuanced relationships in the multi-class classification task. The results suggest that the PLA could benefit from a more sophisticated model, such as a multi-layer perceptron (MLP), that is better suited to handle the intricacies of multi-class classification, especially for digits with similar visual features or those that are less distinct.
   - Additionally, addressing class imbalances through methods like class weighting, data augmentation, or more advanced regularization techniques could further improve the model’s performance, especially for underperforming classes like digit 8.

In conclusion, the PLA demonstrates surprisingly solid performance overall, but there is significant variability across different digit classes, with certain digits being more challenging for the model to classify correctly. Implementing a more flexible model, addressing data imbalances, or refining the decision boundaries could lead to better generalization and accuracy across all digit classes.

**Learning Curves**<br>
At this point, we will generate visualizations of the results. Plotting allows us to better understand the distribution of data, model performance, and any patterns or insights that may not be immediately apparent in the raw data.
"""

# Plot learning curves
plot_learning_curves(pla_model, pla_model_details)

"""When analyzing the graph we can observe that as training progresses, the error rates plateau, with minimal differences observed between training and test errors throughout the epochs. This suggests that the model has learned effectively up to a certain point. However, the minimal reduction in error after a certain number of iterations indicates that the model reaches a saturation point where further learning does not significantly improve performance. This plateau in error reduction may be due to the inherent limitations of the Perceptron Learning Algorithm (PLA), which is a relatively simple and linear model. The close proximity of the training and test error curves is particularly notable, as typically one expects a slight difference, with training error being marginally lower due to the model's direct optimization on the training set.

We can also see that the model’s performance on the test set almost mirrors its performance on the training set, which could reflect the PLA’s inability to capture the complexities of the multi-class dataset. The one-vs-all classification strategy employed by the model may struggle with issues such as class imbalance and the nuanced decision boundaries that are often present in multi-class problems. Moreover, the PLA’s binary decision-making nature, while suitable for simpler tasks, may fail to capture the intricacies and uncertainties inherent in multi-class classification, leading to similar error rates for both training and test sets.

Given these factors, the lack of significant difference between the training and test error curves is likely not indicative of overfitting, but rather a result of the PLA's simplicity and limited capacity to adapt to the complexity of the data. A more flexible and sophisticated model, such as a multi-class perceptron or a model with more advanced decision-making capabilities, might offer a better understanding of the data and potentially improve performance by capturing more complex patterns. Thus, while the observed convergence is a sign of the model reaching its learning limit, the close alignment of the error rates is primarily a reflection of the PLA’s limitations in handling complex, multi-class datasets.

**Confusion Matrix**<br>
We will examine the confusion matrix to gain insights into the classification accuracy and the model's ability to distinguish between different classes.
"""

# Plot the model's confusion matrix
plot_confusion_matrix(y_test_orig, pla_y_pred, pla_model_details)

"""
The confusion matrix reveals interesting patterns about the classifier's performance on the MNIST dataset. As anticipated from a linear classifier lacking the hierarchical feature extraction capabilities of neural networks, the model exhibits notable confusion between visually similar digit pairs. Particularly pronounced misclassifications occur between digits that share structural similarities: there are many of instances where the digit '4' is misclassified as '9', and also cases where '9' is incorrectly labeled as '4', likely due to their similar upper curves and vertical strokes. The model also demonstrates significant confusion between '3' and '5', with many cases of '5' being misclassified as '3' or '8', presumably due to their shared curved segments. Moreover, the perceptron shows weakness in distinguishing '7' from '1', with some cases of '7' being misclassified as '9', reflecting the challenge of discriminating between digits with strong vertical components using only linear decision boundaries.

These patterns of misclassification align with the inherent limitations of the perceptron architecture, which lacks the capacity to learn complex, nonlinear decision boundaries necessary for more nuanced visual feature discrimination.

#### **Model: Multiclass PLA V1 - Results Discussion**

The implementation of the model for the classification task demonstrates both significant strengths and notable limitations. The model achieves an overall accuracy of **83.6%**, with particularly strong performance in recognizing digits 1, 6, and 7. However, the model shows difficulties with digits 5 and 8.

##### **Training and Test Error Trends:**
   - **Training Error:**
     Over the 300 epochs, the training error consistently decreases from 18.26% to 4.17%, reflecting the model’s ability to learn from the data. The relatively steady decrease in training error suggests that the perceptron is effectively minimizing its error on the training set.
   
   - **Test Error:**
     The test error fluctuates but gradually decreases from 37.22% to 5.76% over the epochs. An interesting pattern occurs between epochs 20-100, where the test error is consistently lower than the training error. This is unusual as one would typically expect training error to be lower. Several factors might contribute to this: potential implementation nuances in the one-vs-all strategy, the binary nature of perceptron decisions, or the distribution of complex cases in the datasets. After epoch 150, the error curves converge, suggesting diminishing returns with additional training.

##### **Accuracy by Class:**

**Strong Performers:**<br>
  Digit 0 (Accuracy: 96.35%) and Digit 1 (Accuracy: 96.00%) show exceptional performance. These digits are correctly classified most of the time, with both achieving accuracy above 95%, suggesting that the perceptron model has learned to distinguish them effectively.
  Digit 6 (Accuracy: 94.77%) also performs well, with a high accuracy indicating that the model is proficient at recognizing this digit.

**Moderate Performers:**<br>
  Digit 2 (Accuracy: 83.91%) and Digit 3 (Accuracy: 84.37%) achieve moderate accuracy, which suggests that while the model performs decently with these digits, there is still room for improvement, especially with more complex handwriting styles or varying slants.
  Digit 4 (Accuracy: 85.17%) also performs reasonably well, though it is more prone to misclassifications compared to digits 0, 1, or 6.
  Digit 7 (Accuracy: 86.96%) shows fairly strong performance, but still lags behind the top-performing digits like 0 and 1.

**Weak Performers**<br>
  Digit 5 (Accuracy: 59.39%) and Digit 8 (Accuracy: 66.18%) have significantly lower accuracies, reflecting that the model struggles more with these digits. The difficulty in recognizing digit 5 is further confirmed by its low recall and F1-score, which likely stems from its visual similarities with other digits (such as 3 and 9).
  Digit 9 (Accuracy: 78.38%) also shows lower performance, although it is still recognized with better accuracy than digits 5 and 8. This suggests that the model may have trouble differentiating between digit 9 and other similar digits like 4 or 3.


##### **Precision, Recall, and F1-Score Metrics:**
   - **Precision:**
     The precision values are generally high across most digits, particularly for digits 1, 6, and 7. However, precision for digit 3 is lower, indicating more misclassifications for this digit.
   
   - **Recall:**
     Recall shows more variation. The model struggles most with digit 5, achieving the lowest recall. Recall for digit 0 and 1 is much higher, indicating that the model is very good at identifying these digits. The challenges with digits 5 and 8 may arise due to visual similarities with other digits, such as the resemblance between 5 and 3.
   
   - **F1-Score:**
     The F1-scores are generally strong, with digit 1 achieving the highest score. However, digits like 5 and 3 have relatively lower F1-scores, reflecting the model’s difficulties in accurately distinguishing these digits.

##### **Confusion Matrix Analysis:**
   The confusion matrix highlights specific challenges in distinguishing visually similar digit pairs, such as 4-9 and 5-3. These pairs are difficult for the perceptron to differentiate due to the linear decision boundary limitations inherent in the perceptron model. The one-vs-all approach, while effective, may not capture the complex relationships between similar digits, leading to misclassifications in such cases.

##### **Learning Curves and Convergence:**
   The learning curves show that the model initially converges rapidly, with both training and test errors decreasing steeply between epochs 20-100. After epoch 150, the learning curves flatten, indicating that the model has largely stabilized. This suggests that additional training beyond this point would yield diminishing returns. The test error remaining below the training error early on, followed by the curves converging, highlights some interesting dynamics of the perceptron model, potentially influenced by the one-vs-all strategy or the binary nature of the decision-making process.

##### **Overall Performance:**
   The model’s overall classification accuracy of above 80% is a solid result for a basic perceptron approach. However, it has room for improvement, particularly with digits like 5 and 8, where precision and recall are lower. The challenges in distinguishing visually similar digits (e.g., 4-9, 5-3) are also indicative of the perceptron’s limitations in handling more complex, non-linear decision boundaries.

##### **Conclusion:**
The perceptron with the one-vs-all strategy is a solid starting point for the MNIST digit classification task, achieving a respectable overall accuracy above 80%. It performs well for most digits, particularly 0, 1, and 6, but struggles with certain digits like 5, 8, and 9, which results in lower classification accuracy for these classes. The difficulties in distinguishing visually similar digits, combined with the linear decision boundaries of the perceptron model, highlight the inherent limitations of this approach. While the perceptron provides a useful baseline, more advanced models (such as neural networks with non-linear activation functions) would likely yield better performance, especially for more complex or overlapping digit classes.
"""