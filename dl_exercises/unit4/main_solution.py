# do not run
from dataset import load_dataset, plot_dataset, preprocess_dataset
from model import build_model, train_model
from evaluate import calc_accuracy, plot_history


# Load CIFAR dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Plot images from dataset
plot_dataset(X_train, y_train)

# Preprocess dataset to enter NN
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_dataset(
    X_train, y_train, X_val, y_val, X_test, y_test)

# Build CNN model as a classifier
model = build_model()

# Train the model
n_epochs = 50
h = train_model(model, X_train, y_train, X_val, y_val, n_epochs)

# Evaluate accuracy obtained by model
calc_accuracy(model, X_test, y_test)

# Plot history from training step
plot_history(h)

# Save the model
save_model(model, path)
