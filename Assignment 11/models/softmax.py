"""
## PART B: SoftMax
 In this section, we'll define the Softmax regression model. Softmax regression calculates the probability of each class by using the softmax function. The softmax function converts a vector of values into a probability distribution, where each value is between 0 and 1, and the sum of all values is 1.


The goal is to estimate the probability $P(y = k | x)$ for each class $k \in \{1, \dots, K\}$, where $K=10$ is the total number of classes. The Softmax function ensures that the predicted probabilities for each class sum to 1. The hypothesis of the model can be expressed as:

$$
h(x) = \left( P(y = 1 | x), P(y = 2 | x), \dots, P(y = K | x) \right)
$$

Where the probabilities are computed using the softmax function applied to the weighted input $x$.


 #### **Define the Softmax Regression Model**
In this model, we aim to minimize the softmax cost function, which is a cross-entropy loss function. The model calculates probabilities for each of the possible classes and selects the class with the highest probability. We will implement the gradient descent method to minimize the loss function.


### **Softmax Function and Loss Computation**
1. **Softmax Function**:
   - The softmax function transforms logits into probabilities:  
     $$ h(x) = \frac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}} $$  
     where $w_k$ is the weight vector for class $k$, and $x$ is the input.

2. **Loss Calculation**:
   - The model minimizes the cross-entropy loss, defined as:  
     $$ E_{\text{in}}(w) = - \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K 1\{y_n = k\} \log P(y = k | x_n) $$  
     where $P(y = k | x_n)$ is the predicted probability for class $k$, and $y_n$ is the true label.

3. **Gradient Calculation**:
   - The gradient of the loss with respect to the weights is computed as:  
     $$ \nabla_w E_{\text{in}} = \frac{1}{N} \sum_{n=1}^N (P(y = k | x_n) - 1\{y_n = k\}) x_n^T $$

4. **Prediction**:
   - For a given input $x$, the predicted class is:  
     $$ \hat{y} = \arg\max_k P(y = k | x) $$

The model is trained using mini-batch gradient descent, updating weights iteratively to minimize the loss function.

### **SoftmaxRegression Model Overview**

This class implements a softmax regression model using the MNIST dataset for digit classification. This model is based on the softmax function, which is commonly used for multi-class classification problems, such as digit recognition in this case. The model is trained using stochastic gradient descent (SGD) with mini-batch updates. It includes mechanisms for training, evaluation, and visualization of results, including accuracy, loss, and confusion matrix plots.

### **Model Implementation**

#### Model Structure and Mechanisms

The class defines the following key components:

1. **Initialization**:
   - `learning_rate`: Sets the learning rate for gradient descent.
   - `num_epochs`: Defines how many times the model will iterate over the entire dataset during training.
   - `batch_size`: Specifies the number of samples per batch during training.

2. **Data Loading and Preprocessing**:
   - The `load_mnist` function fetches and preprocesses the MNIST dataset, normalizing the pixel values to the range [0, 1] and adding a bias term. The dataset is split into training and testing sets.

3. **Model Mechanics**:
   - **One-Hot Encoding**: Converts the target labels into a one-hot encoded format.
   - **Softmax Function**: Converts raw model outputs (logits) into probability distributions for each class.
   - **Loss Calculation**: Uses the cross-entropy loss function to quantify the model's performance.
   - **Gradient Calculation**: Computes gradients for weight updates using backpropagation.
   - **Prediction**: The model predicts the class with the highest probability for each input.

4. **Training Process**:
   - The `train` method updates model weights using mini-batch gradient descent.
   - The loss for each epoch is computed, and the model's performance on the test set is evaluated after each epoch.

5. **Evaluation**:
   - **Accuracy**: The model's prediction accuracy is calculated.
   - **Confusion Matrix**: Generates a confusion matrix to assess performance for each class (digit).
   - **Sensitivity**: Calculates the sensitivity (True Positive Rate) for each class.

6. **Visualization**:
   - **Loss Curves**: The `plot_loss` method visualizes the training and test loss over epochs.
   - **Confusion Matrix**: The `plot_confusion_matrix` method displays a heatmap of the confusion matrix.

#### **Code Summary**

The `SoftmaxRegression` class includes:
- `__init__`: Initializes hyperparameters and model parameters.
- `load_mnist`: Loads and preprocesses the MNIST dataset.
- `one_hot_encode`: Converts labels to one-hot encoding.
- `softmax`: Applies the softmax function to logits.
- `compute_loss`: Computes the cross-entropy loss.
- `compute_gradient`: Computes the gradient for backpropagation.
- `predict`: Predicts the class for input data.
- `train`: Trains the model using mini-batch gradient descent.
- `calculate_accuracy`: Computes the accuracy of the model.
- `calculate_confusion_and_sensitivity`: Computes the confusion matrix and sensitivity for each class.
- `plot_confusion_matrix`: Plots the confusion matrix as a heatmap.
- `plot_loss`: Plots the training and test loss curves.
- `run_model`: Executes the full workflow, including training, prediction, evaluation, and visualization.

#### **Reasoning Behind Design Choices**

1. **Softmax for Multi-class Classification**:
   Softmax regression is chosen because it is a natural fit for multi-class classification problems, like digit recognition in MNIST, where each input belongs to one of ten possible classes.

2. **Stochastic Gradient Descent (SGD)**:
   The model uses SGD with mini-batches to balance computational efficiency and convergence speed. This allows the model to process large datasets like MNIST effectively.

3. **Cross-Entropy Loss**:
   Cross-entropy loss is used because it is the standard loss function for classification problems involving probabilities and softmax outputs.

4. **Evaluation with Confusion Matrix and Sensitivity**:
   The confusion matrix and sensitivity are used to evaluate how well the model performs for each class, providing more detailed insights into its strengths and weaknesses.

#### **Approach Summary**

This model follows a straightforward approach to multi-class classification using softmax regression. The training is done through mini-batch gradient descent, and the model is evaluated using a combination of accuracy, confusion matrix, and sensitivity metrics. The use of visualization tools for the loss curves and confusion matrix enhances the interpretability and understanding of the model’s performance.

**Note:**<br>
The `load_mnist` function is provided for convenience, but the data remains identical to that mentioned previously.
"""

"""
Algorithm Overview:
-----------------
Softmax regression extends logistic regression to handle multiple classes by using the softmax function
to convert raw model outputs into probability distributions over the classes.

Key Components:
1. Softmax Function: σ(z)_i = exp(z_i) / Σ_j exp(z_j)
   - Converts raw scores into probabilities that sum to 1
   - Numerically stabilized by subtracting max value before exponential

2. Cross-Entropy Loss: -Σ y_i log(p_i)
   - Measures difference between predicted probabilities and true labels
   - Using one-hot encoded ground truth labels

3. Gradient Descent:
   - Computes gradient of loss with respect to weights
   - Updates weights iteratively: w = w - η∇L
   - Uses mini-batches for better computational efficiency
"""
class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, num_epochs=50, batch_size=128):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weights = None

    def load_mnist(self):
        """
        Loads and preprocesses the MNIST dataset.

        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test)
            - X_train, X_test: Features with added bias term and normalized pixels
            - y_train, y_test: Integer labels for digits
        """
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy().astype(int)
        X = X / 255.0
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def one_hot_encode(y, num_classes=10):
        return np.eye(num_classes)[y]

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, X, y_encoded):
        z = np.dot(X, self.weights.T)
        probs = self.softmax(z)
        return -np.mean(np.sum(y_encoded * np.log(probs + 1e-10), axis=1))

    def compute_gradient(self, X, y_encoded):
        z = np.dot(X, self.weights.T)
        probs = self.softmax(z)
        grad = np.dot(X.T, (probs - y_encoded)).T / X.shape[0]
        return grad

    def predict(self, X):
        z = np.dot(X, self.weights.T)
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    def train(self, X_train, y_train, X_test, y_test):
        num_features = X_train.shape[1]
        num_classes = 10
        self.weights = np.random.randn(num_classes, num_features) * 0.01

        y_train_encoded = self.one_hot_encode(y_train)
        y_test_encoded = self.one_hot_encode(y_test)

        num_batches = len(X_train) // self.batch_size
        train_losses = []
        test_losses = []

        progress_bar = tqdm(
            range(self.num_epochs),
            desc="Training Progress",
            unit="epoch",
            leave=False,
            position=0,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )

        for epoch in progress_bar:
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train_encoded[indices]

            epoch_loss = 0
            for batch in range(num_batches):
                start_idx = batch * self.batch_size
                end_idx = start_idx + self.batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                grad = self.compute_gradient(X_batch, y_batch)
                self.weights -= self.learning_rate * grad

                batch_loss = self.compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)

            test_loss = self.compute_loss(X_test, y_test_encoded)
            test_losses.append(test_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

        return train_losses, test_losses

    def calculate_confusion_and_sensitivity(self, y_true, y_pred):
        """
        Computes the confusion matrix for each digit and sensitivity (TPR) for each class.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels

        Returns:
        --------
        tuple: (confusion_matrix, sensitivities)
            confusion_matrix : numpy.ndarray
                Confusion matrix for the classification results.
            sensitivities : list
                Sensitivity values for each digit (TPR).
        """
        cm = confusion_matrix(y_true, y_pred, num_classes=10)
        sensitivities = []
        for i in range(10):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sensitivities.append(sensitivity)
        return cm, sensitivities

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, num_classes=10, num_epochs=50, lr=0):
        """
        Plots a confusion matrix to evaluate classification performance.

        Args:
            y_true (numpy.ndarray): True labels of the dataset.
            y_pred (numpy.ndarray): Predicted labels by the model.
            model_details (dict, optional): Details about the model for annotation.
            num_classes (int, optional): Number of classes for confusion matrix.
        """
        model_details = {
            'model_name': 'Softmax Regression',
            'learning_rate': lr,
            'epochs': num_epochs
            }

        sns.set_theme(style="whitegrid")

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, num_classes=10)

        # Calculate model accuracy
        accuracy = np.mean(y_true == y_pred)

        # Initialize figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create a heatmap for the confusion matrix
        sns.heatmap(cm, cmap='RdPu', square=True, linewidth=0.3, annot=True, fmt='d',
                    annot_kws={'size': 12})

        # Add titles and labels
        plt.title('Classification Results\n' +
                  f'Accuracy: {accuracy:.2%}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        # Display the model details
        if model_details:
            detail_text = (
                r"$\bf{Model:}$" + f" {model_details.get('model_name', 'N/A')}\n" +
                r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n" +
                r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
            )
            fig.text(
                0.02, 0.98, detail_text,
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.3', alpha=0.9)
            )

        # Enhance visual appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Show the plot
        plt.show()

    @staticmethod
    def calculate_sensitivity(y_true, y_pred):
        """
        Calculates sensitivity (TPR) for each class.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels

        Returns:
        --------
        list: Sensitivity values for each digit
        """
        cm = confusion_matrix(y_true, y_pred, num_classes=10)
        sensitivities = []
        for i in range(10):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn)
            sensitivities.append(sensitivity)
        return sensitivities


    def plot_loss(self, train_losses, test_losses, model_details=None, num_epochs=50, lr=0):
        """
        Visualizes the training and test loss over epochs with a customized style.

        Args:
            train_losses: List of training loss values.
            test_losses: List of test loss values.
            model_details (dict, optional): Additional details about the model, such as
                                            name and number of epochs, for annotation.
        """
        model_details = {
            'model_name': 'Softmax Regression',
            'learning_rate': lr,
            'epochs': num_epochs
            }
        # Set a custom color palette and grid-based visual theme
        color_palette = {
            'train': '#4B0082',  # Indigo for training loss
            'test': '#FF6347'    # Tomato for test loss
        }
        sns.set_theme(style="whitegrid")  # Apply a white grid theme for improved readability

        # Initialize the figure and axis with dimensions
        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate the epochs range
        epochs = range(1, len(train_losses) + 1)

        # Plot training and test loss curves
        ax.plot(epochs, train_losses, label='Training Loss',
                linewidth=2, marker='.', markersize=4, color=color_palette['train'], alpha=0.8)
        ax.plot(epochs, test_losses, label='Test Loss',
                linewidth=2, marker='.', markersize=4, color=color_palette['test'], alpha=0.8)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit number of x-axis ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Limit number of y-axis ticks

        # Calculate average test loss to include in the title
        avg_test_loss = np.mean(test_losses)

        # Add a descriptive title and labels
        ax.set_title('Training and Test Loss over Epochs\n' +
                    f'Average Test Loss: {avg_test_loss:.3f}',
                    fontsize=16, fontweight='bold', color='#333333', pad=20)
        ax.set_xlabel('Epochs', fontsize=12, color='#555555')
        ax.set_ylabel('Loss', fontsize=12, color='#555555')

        # Add legend
        ax.legend(frameon=True, facecolor='white',
                  edgecolor='lightgray', loc='upper right')

        # Customize the background colors
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Display the model details
        if model_details:
            detail_text = (
                r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'N/A')}\n"
                r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n" +
                r"$\bf{Epochs:}$" + f" {model_details.get('epochs', len(epochs))}"
            )
            fig.text(
                0.02, 0.98, detail_text,
                ha='left', va='top', fontsize=10,
                color='#333333',
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.4', alpha=0.9),
                usetex=False
            )

        # Adjust layout to prevent overlap and display the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def run_model(self):
        X_train, X_test, y_train, y_test = self.load_mnist()
        train_losses, test_losses = self.train(X_train, y_train, X_test, y_test)

        y_pred = self.predict(X_test)
        test_accuracy = self.calculate_accuracy(y_test, y_pred)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        conf_matrix, sensitivities = model.calculate_confusion_and_sensitivity(y_test, y_pred)

        self.plot_loss(train_losses, test_losses, num_epochs=self.num_epochs, lr=self.learning_rate)
        self.plot_confusion_matrix(y_test, y_pred, num_epochs=self.num_epochs)

        # sensitivities = self.calculate_sensitivity(y_test, y_pred)
        # for digit, sens in enumerate(sensitivities):
        #     print(f"Sensitivity for digit {digit}: {sens:.4f}")

        cm, sensitivities = self.calculate_confusion_and_sensitivity(y_test, y_pred)
        print("\nConfusion Matrix:")
        # print(cm) # print confusion matrix without a plot
        print("\nSensitivities (TPR) for each digit:")
        for digit, sens in enumerate(sensitivities):
            print(f"Digit {digit}: {sens:.4f}")

# Run model to observe initial performance; further analysis will follow later
model = SoftmaxRegression(learning_rate=0.001, num_epochs=100)
model.run_model()

"""
We want to evaluate a broader range of hyperparameters by selecting configurations with more epochs, allowing us to see how the model performs over extended training periods. For instance, in some of the configurations, we'll increase the number of epochs to 100. By doing so, we can observe how the model improves over time, and more importantly, we can see if good results are already achievable in the early stages of training. This is useful because it helps us determine if the model is converging too slowly or too quickly, and whether further training beyond a certain point is necessary.

Additionally, we want to explore a wide range of learning rates, from 0.01 to 0.2. The reason for this is that the learning rate significantly influences how quickly the model converges. A higher learning rate may cause the model to converge faster, but it can also make the training unstable, potentially leading to overshooting the optimal solution. On the other hand, a lower learning rate will result in slower convergence, allowing the model to make smaller, more stable adjustments. By testing a wide range of learning rates, we aim to strike a balance between training time and accuracy, and ensure we don't miss the optimal learning rate that provides the best results for this task.

By running experiments with more epochs and a variety of learning rates, we can better understand how the model performs both in the early stages and as it continues to train. This also allows us to pinpoint the most effective learning rate for stable and efficient training, while ensuring that the model converges at the optimal point within the training process.

We evaluate and compare the performance of several models using different hyperparameter configurations. Specifically, we will vary the learning rate, number of epochs, and batch size across four different setups. For each configuration, the model will be trained on the MNIST dataset, and we will calculate key performance metrics such as training and test losses, accuracy, and class-wise sensitivities. Sensitivity, or True Positive Rate (TPR), helps us understand how well the model identifies each digit class. After training, we will visualize the results, including loss curves, confusion matrices, and sensitivities. Finally, we will compare the models based on their overall accuracy, average training and test losses, and sensitivities to determine the optimal hyperparameter configuration for the task.

#### **Configuration 1:**
- **Learning Rate:** 0.1  
- **Epochs:** 100  
- **Batch Size:** 128  

In this configuration, we aim to balance speed and stability. A learning rate of 0.1 is moderate, which means the model can make relatively large updates to weights, facilitating faster convergence while reducing the risk of overshooting the optimal solution. The 100 epochs provide ample time for the model to improve and refine its parameters. By using this configuration, we can observe the model’s performance at different stages of training, ensuring that good results are visible early on. The batch size of 128 is a typical choice, balancing computational efficiency and model performance, allowing the model to update its weights in manageable steps.

#### **Configuration 2:**
- **Learning Rate:** 0.05  
- **Epochs:** 100  
- **Batch Size:** 64  

This configuration uses a lower learning rate (0.05), which will slow down the rate of convergence. This setup allows for more precise weight updates, preventing the model from making large, unstable steps in parameter space. With 100 epochs, this configuration gives the model enough time to adjust more finely to the data, providing a good balance between training time and model accuracy. The batch size of 64 is smaller than the previous configuration, meaning the model will perform more frequent weight updates. While this increases training time per epoch, it can help avoid potential instability from larger batch sizes and allow for a more detailed exploration of the parameter space.

#### **Configuration 3:**
- **Learning Rate:** 0.01  
- **Epochs:** 100  
- **Batch Size:** 128  

This configuration uses an even smaller learning rate of 0.01, which results in slower but more stable convergence. By taking smaller steps during training, the model can make finer adjustments to weights, which could lead to a better generalization of the model. The number of epochs (100) allows for extended training, ensuring that the model can further refine its weights over a longer period. This configuration is particularly useful for exploring whether a slower learning process leads to better performance. The batch size of 128 strikes a balance between computational efficiency and the frequency of weight updates, making it a good choice for stability during training.

#### **Configuration 4:**
- **Learning Rate:** 0.2  
- **Epochs:** 50  
- **Batch Size:** 256  

In this configuration, we choose a larger learning rate (0.2), which will lead to faster convergence but may also cause instability in the learning process. This configuration tests the model's ability to handle rapid learning and still converge to a good solution. The number of epochs is set to 50, fewer than in other configurations, to compensate for the faster learning rate. Fewer epochs are generally needed when the model learns quickly, as long as the learning rate is set appropriately. The batch size of 256 is relatively large, meaning fewer updates per epoch but more data per update. This setup could potentially speed up the training process, allowing the model to learn from more data at once, but may reduce the precision of each weight update. This configuration is useful for determining if faster convergence can still lead to good results and if larger batch sizes improve training efficiency.

#### **Why These Configurations?**

We chose to explore a wide range of learning rates to understand the model’s sensitivity to different step sizes during training. Learning rates that are too high might cause the model to overshoot the optimal solution, resulting in poorer performance, while learning rates that are too low may lead to slower convergence and potentially longer training times. By examining a range of values—from 0.01 to 0.2—we aim to find the optimal learning rate that balances speed and stability.

Furthermore, the choice of 100 epochs in most configurations ensures that the models have sufficient time to learn and refine their weights. While early-stage performance might already provide insights into the effectiveness of a configuration, having more epochs allows us to observe whether improvements continue beyond the early stages of training. This will help us determine whether additional training time results in diminishing returns or if it allows the model to achieve better performance.

In summary, by testing these four configurations, we can assess the impact of different learning rates, batch sizes, and epoch counts on the model's ability to learn and generalize from the MNIST dataset. This will provide valuable insights into the trade-offs between training speed, model stability, and accuracy, enabling us to identify the most effective hyperparameter settings for the task.
"""

# Define a list of different hyperparameter configurations to test
# Each configuration is a dictionary containing the learning rate, number of epochs, and batch size
configs = [
    {"learning_rate": 0.1, "num_epochs": 100, "batch_size": 128},  # Configuration 1
    {"learning_rate": 0.05, "num_epochs": 100, "batch_size": 64},  # Configuration 2
    {"learning_rate": 0.01, "num_epochs": 100, "batch_size": 128},  # Configuration 3
    {"learning_rate": 0.2, "num_epochs": 50, "batch_size": 256}  # Configuration 4
]

# Initialize an empty dictionary to store the results of each configuration for comparison
results = {}

progress_bar = tqdm(
    total=len(configs),  # Total number of iterations
    desc="Processing Configurations",  # Description of the task
    unit="config",  # Unit of measurement
    leave=True,  # Leave the progress bar displayed after completion
    position=0,  # Position of the progress bar (useful in multi-threaded applications)
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# Iterate through each configuration and run the model with those hyperparameters
for idx, config in enumerate(configs):
    progress_bar.set_description(f"Processing Model {idx + 1} with config: {config}")  # Update description dynamically

    print(f"\nRunning Model {idx + 1} with config: {config}")  # Log the configuration being tested
    model = SoftmaxRegression(
        learning_rate=config["learning_rate"],  # Set the learning rate
        num_epochs=config["num_epochs"],  # Set the number of epochs
        batch_size=config["batch_size"]  # Set the batch size
    )

    # Load the MNIST dataset
    X_train, X_test, y_train, y_test = model.load_mnist()

    # Train the model and get the training and test losses
    train_losses, test_losses = model.train(X_train, y_train, X_test, y_test)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = model.calculate_accuracy(y_test, y_pred)

    # Calculate the confusion matrix and sensitivities
    conf_matrix, sensitivities = model.calculate_confusion_and_sensitivity(y_test, y_pred)

    # Store the results of the current configuration
    results[idx] = {
        "config": config,  # Store the configuration used
        "train_losses": train_losses,  # Store the training losses
        "test_losses": test_losses,  # Store the test losses
        "accuracy": accuracy,  # Store the accuracy
        "confusion_matrix": conf_matrix,  # Store the confusion matrix
        "sensitivities": sensitivities  # Store the sensitivities
    }

    # Log the accuracy and sensitivities of the current model
    print(f"Accuracy for Model {idx + 1}: {accuracy:.4f}")
    print(f"Sensitivities for Model {idx + 1}: {['{:.4f}'.format(s) for s in sensitivities]}")

    # Plot the loss curves and confusion matrix for the current model
    model.plot_loss(train_losses, test_losses, num_epochs=config["num_epochs"], lr=config["learning_rate"])
    model.plot_confusion_matrix(y_test, y_pred, num_epochs=config["num_epochs"], lr=config["learning_rate"])
    progress_bar.update(1)

# Compare the results of all models
print("\nComparison of Models:")
for idx, result in results.items():
    config = result["config"]  # Get the configuration of the model
    accuracy = result["accuracy"]  # Get the accuracy of the model
    avg_train_loss = np.mean(result["train_losses"])  # Calculate the average training loss
    avg_test_loss = np.mean(result["test_losses"])  # Calculate the average test loss
    sensitivities = result["sensitivities"]  # Get the sensitivities of the model

    # Log the comparison results
    print(f"Model {idx + 1} | Config: {config} | Accuracy: {accuracy:.4f} | "
          f"Avg Train Loss: {avg_train_loss:.4f} | Avg Test Loss: {avg_test_loss:.4f}")
    print(f"   Sensitivities: {[f'{s:.4f}' for s in sensitivities]}")

"""#### Performance Overview

All models demonstrate strong performance, with high classification accuracy across the board with values ranging above 90%. However, differences emerge in their respective training and test losses, as well as in the sensitivity of the models to individual digit classes.

#### Comparison of Hyperparameters

- **Learning Rate**: Models with higher learning rates generally perform comparably in terms of accuracy, though some slight differences in test loss are observed. The highest learning rate tends to allow faster convergence, but it can lead to larger fluctuations in performance across epochs. Conversely, lower learning rates, while leading to slower convergence, tend to result in higher training and test losses.

- **Epochs and Batch Size**: The number of epochs and batch size also significantly impact performance. Models with larger batch sizes and fewer epochs tend to achieve better results more efficiently, balancing both training time and model accuracy. On the other hand, models that use smaller batch sizes and more epochs tend to have more gradual improvements but can also suffer from longer training times and potentially higher losses.

#### Conclusion

The Softmax Regression models, in general, perform well on the MNIST dataset, demonstrating solid classification accuracy. Sensitivity levels for most digits are high, hovering around 90%, which indicates that the models are effectively identifying the majority of digit classes. However, slight variations in accuracy and loss arise based on hyperparameter configurations, such as learning rate, batch size, and the number of epochs.

Despite these variations, the changes in model performance resulting from adjustments to these hyperparameters are negligible. This can be attributed to several factors:

1. **Simplicity of Softmax Regression**: Softmax Regression is a relatively simple linear model compared to more complex architectures like neural networks. As such, it may have limited sensitivity to hyperparameter changes. In other words, it doesn't require extensive tuning of hyperparameters to achieve high performance on tasks like MNIST, which is a relatively simple problem. The model has already captured the necessary patterns in the data, and small adjustments in hyperparameters don't significantly affect its performance.

2. **Dataset Characteristics**: The MNIST dataset is well-known and relatively simple, with clear patterns in the handwritten digits that make it an ideal benchmark for classification models. The features in MNIST are straightforward and don't require intricate hyperparameter tuning to achieve high performance. As a result, even with different hyperparameter configurations, the model can still generalize well to unseen data without substantial changes in its predictive capabilities.

3. **Saturation of Model Performance**: Given that the MNIST task is a relatively low-complexity classification problem, the model performance quickly reaches a plateau once it has learned the basic patterns in the data. The Softmax Regression models are already quite effective at classifying the digits, so small adjustments to the learning rate, batch size, or number of epochs don't yield significant improvements. Once the model has converged to a good solution, additional tuning typically leads to diminishing returns.

4. **Hyperparameter Interactions**: In this case, the adjustments to hyperparameters—such as learning rate or batch size—might be compensating for each other, leading to similar overall performance. For example, while increasing the learning rate may allow the model to converge faster, it might also increase variance in the gradient updates. However, when paired with a larger batch size, this can stabilize the learning process, leading to similar performance to models that use smaller learning rates but more epochs. These compensating effects mean that the model’s final performance remains relatively consistent across different configurations.

Therefore, the lack of significant performance differences across hyperparameter settings is a result of the combination of the simplicity of the Softmax Regression model, the characteristics of the MNIST dataset, and the fact that the model has already learned to classify the digits well even with suboptimal configurations. For further improvements, more complex models or advanced techniques would likely be required.
"""