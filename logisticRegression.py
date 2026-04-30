import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#logistic regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iters=300):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        for epoch in range(self.num_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute training loss
            train_loss = -np.mean(
                y * np.log(y_pred + 1e-9) +
                (1 - y) * np.log(1 - y_pred + 1e-9)
            )

            # Compute training accuracy
            train_preds = (y_pred >= 0.5).astype(int)
            train_acc = np.mean(train_preds == y)

            # Store training metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation metrics (if provided)
            if X_val is not None and y_val is not None:
                val_linear = np.dot(X_val, self.weights) + self.bias
                val_pred = self.sigmoid(val_linear)

                val_loss = -np.mean(
                    y_val * np.log(val_pred + 1e-9) +
                    (1 - y_val) * np.log(1 - val_pred + 1e-9)
                )

                val_preds = (val_pred >= 0.5).astype(int)
                val_acc = np.mean(val_preds == y_val)

                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Monitor loss
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}")

    def predict_probabilities(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        probabilities = self.predict_probabilities(X)
        return [1 if p >= 0.5 else 0 for p in probabilities]

#function to split the data
def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Split X, y into train/validation/test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_size), random_state=random_state
    )
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

#function to train the model
def train_model(X_train, y_train, X_val, y_val, learning_rate=0.1, num_iters=300):
    """Initialize and train the logistic regression model."""
    model = LogisticRegression(learning_rate=learning_rate, num_iters=num_iters)
    model.fit(X_train, y_train, X_val, y_val)
    return model

#function to evaluate the model
def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Predict and print accuracy and classification report."""
    y_pred = model.predict(X)
    print(f"\n{dataset_name} Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"{dataset_name} Classification Report:\n{classification_report(y, y_pred)}")
    return y_pred

#function to run the full pipeline
def run_full_workflow(X, y, learning_rate=0.1, num_iters=300):
    """Split data, train model, evaluate on validation and test sets."""
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    model = train_model(X_train, y_train, learning_rate, num_iters)

    print("\n--- Validation Set ---")
    evaluate_model(model, X_val, y_val, dataset_name="Validation")

    print("\n--- Test Set ---")
    evaluate_model(model, X_test, y_test, dataset_name="Test")

    return model, (X_train, y_train, X_val, y_val, X_test, y_test)