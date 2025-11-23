"""
Neural Networks Starter Exercise
Week 10 - Advanced Programming 2025

Learning Goals:
1. Build and train neural networks with Keras
2. Understand the difference between binary and multi-class classification
3. Debug common neural network issues
4. Experiment with architecture and hyperparameters

Time: 45 minutes
- Part 1: Binary Classification (15 min)
- Part 2: Multi-class Classification (15 min)
- Part 3: Debugging Challenge (15 min)
- Bonus: Advanced techniques (if time permits)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_iris, load_breast_cancer

# Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ Please install TensorFlow: pip install tensorflow")
    exit(1)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("PART 1: BINARY CLASSIFICATION - Cancer Detection")
print("=" * 60)

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")

# TODO 1: Split the data (80/20 train/test split)
# Hint: Use train_test_split with random_state=42
X_train, X_test, y_train, y_test = None, None, None, None

# TODO 2: Scale the features
# Hint: Neural networks work better with scaled data
# Use StandardScaler
scaler = None
X_train_scaled = None
X_test_scaled = None

# TODO 3: Build a binary classification model
# Architecture suggestion:
# - Input layer matching X_train dimensions
# - Hidden layer 1: 32 neurons, relu activation
# - Hidden layer 2: 16 neurons, relu activation
# - Output layer: ??? neurons, ??? activation (think: binary classification!)

model_binary = Sequential([
    # TODO: Add layers here
])

# TODO 4: Compile the model
# Hint: Binary classification needs:
# - optimizer: 'adam'
# - loss: ??? (what loss for binary classification?)
# - metrics: ['accuracy']

model_binary.compile(
    optimizer=None,
    loss=None,
    metrics=None
)

# TODO 5: Train the model
# Train for 50 epochs with validation_split=0.2
# Save the history object to plot later

history_binary = None

# TODO 6: Evaluate on test set
test_loss, test_accuracy = None, None
print(f"\n📊 Test Accuracy: {test_accuracy:.4f}")

# TODO 7: Plot training history
# Plot both loss and accuracy (train vs validation)

def plot_history(history, title="Training History"):
    """Plot training and validation metrics."""
    # TODO: Create 2 subplots (loss and accuracy)
    pass

plot_history(history_binary, "Binary Classification Training")


print("\n" + "=" * 60)
print("PART 2: MULTI-CLASS CLASSIFICATION - Iris Flowers")
print("=" * 60)

# Load iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # 0, 1, 2 (three species)

print(f"Dataset shape: {X_iris.shape}")
print(f"Classes: {np.unique(y_iris)}")
print(f"Number of classes: {len(np.unique(y_iris))}")

# Split and scale
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# TODO 8: Convert labels to one-hot encoding
# Hint: Use keras.utils.to_categorical
# Example: [0, 1, 2] -> [[1,0,0], [0,1,0], [0,0,1]]

y_train_iris_onehot = None
y_test_iris_onehot = None

print(f"Original label shape: {y_train_iris.shape}")
print(f"One-hot label shape: {y_train_iris_onehot.shape}")

# TODO 9: Build a multi-class classification model
# Architecture suggestion:
# - Input layer: 4 features
# - Hidden layer 1: 16 neurons, relu
# - Hidden layer 2: 8 neurons, relu
# - Output layer: ??? neurons, ??? activation (think: multi-class!)

model_multiclass = Sequential([
    # TODO: Add layers here
])

# TODO 10: Compile the model
# Hint: Multi-class needs:
# - optimizer: 'adam'
# - loss: ??? (what loss for multi-class with one-hot encoding?)
# - metrics: ['accuracy']

model_multiclass.compile(
    optimizer=None,
    loss=None,
    metrics=None
)

# TODO 11: Train the model
history_multiclass = None

# TODO 12: Evaluate
test_loss_iris, test_accuracy_iris = None, None
print(f"\n📊 Test Accuracy: {test_accuracy_iris:.4f}")

# TODO 13: Make predictions and show confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred_probs = None  # TODO: Get predictions (probabilities)
y_pred = None  # TODO: Convert to class labels (use np.argmax)

print("\n📈 Confusion Matrix:")
print(confusion_matrix(y_test_iris, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test_iris, y_pred, target_names=iris.target_names))


print("\n" + "=" * 60)
print("PART 3: DEBUGGING CHALLENGE - Fix This Broken Model!")
print("=" * 60)

# This model has MULTIPLE bugs. Find and fix them!
# Hints: Check the loss function, activation function, and data preparation

X_debug, y_debug = make_classification(
    n_samples=1000, n_features=10, n_classes=3,
    n_informative=8, n_redundant=0, random_state=42
)

X_train_debug, X_test_debug, y_train_debug, y_test_debug = train_test_split(
    X_debug, y_debug, test_size=0.2, random_state=42
)

# BUG ALERT: This model will fail or perform poorly!
broken_model = Sequential([
    layers.Dense(32, activation='relu', input_shape=(10,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='sigmoid')  # 🐛 BUG 1: Wrong activation?
])

broken_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # 🐛 BUG 2: Wrong loss?
    metrics=['accuracy']
)

# 🐛 BUG 3: Should we prepare y_train_debug differently?
print("\n🐛 Training broken model...")
try:
    history_debug = broken_model.fit(
        X_train_debug, y_train_debug,
        epochs=20, validation_split=0.2, verbose=0
    )
    print("Model trained, but check the accuracy...")
    test_loss_debug, test_accuracy_debug = broken_model.evaluate(
        X_test_debug, y_test_debug, verbose=0
    )
    print(f"Test Accuracy: {test_accuracy_debug:.4f}")
    print("❓ Is this accuracy suspiciously low? Fix the bugs!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Fix the bugs above to make this work!")

# TODO 14: Create a FIXED version of the model
fixed_model = Sequential([
    # TODO: Fix the architecture
])

fixed_model.compile(
    # TODO: Fix the compilation
)

# TODO: Fix the data preparation if needed
# y_train_debug_fixed = ???

# TODO: Train the fixed model
# history_fixed = fixed_model.fit(...)


print("\n" + "=" * 60)
print("BONUS CHALLENGES")
print("=" * 60)
print("""
Choose one or more bonus challenges:

BONUS 1: Early Stopping (⭐)
- Add EarlyStopping callback to prevent overfitting
- Monitor validation loss and stop when it stops improving
- Compare results with/without early stopping

BONUS 2: Learning Rate Experiments (⭐)
- Try different learning rates: 0.001, 0.01, 0.1
- Plot how learning rate affects training speed and final accuracy
- Hint: Use keras.optimizers.Adam(learning_rate=0.01)

BONUS 3: Regularization (⭐⭐)
- Add Dropout layers to prevent overfitting
- Try different dropout rates (0.2, 0.5)
- Add L2 regularization to Dense layers
- Compare validation accuracy with/without regularization

BONUS 4: Architecture Search (⭐⭐)
- Try different numbers of layers (1, 2, 3, 4)
- Try different layer sizes (8, 16, 32, 64, 128)
- Plot how architecture affects performance
- Which architecture is best for the iris dataset?

BONUS 5: Activation Function Comparison (⭐⭐)
- Compare relu, tanh, sigmoid in hidden layers
- Which works best? Why?
- Visualize the activation functions

BONUS 6: Batch Size Matters (⭐⭐)
- Train with batch_size=8, 32, 128
- Plot training time and final accuracy
- What's the trade-off?

BONUS 7: Save and Load Models (⭐)
- Save your best model: model.save('best_model.keras')
- Load it back: keras.models.load_model('best_model.keras')
- Verify it gives same predictions

BONUS 8: Neural Network Visualization (⭐⭐⭐)
- Visualize what each layer learns
- Plot activation outputs for each layer
- Use keras.Model to create intermediate layer models
- Example:
    layer_model = keras.Model(inputs=model.input,
                             outputs=model.layers[0].output)
    activations = layer_model.predict(X_test[:1])

BONUS 9: Custom Callback (⭐⭐⭐)
- Create a custom callback that prints a message every 10 epochs
- Track and plot the learning rate over time
- Implement a callback that saves model when accuracy > 0.95

BONUS 10: Transfer Learning Lite (⭐⭐⭐)
- Train a model on the full breast cancer dataset
- Freeze the first layer
- Re-train on a subset (100 samples)
- Compare with training from scratch on the subset
- Does "transfer learning" help even with simple models?

BONUS 11: Predict Probabilities (⭐)
- Get probability predictions for test samples
- Find the most confident predictions
- Find the least confident (most uncertain) predictions
- Plot probability distributions

BONUS 12: Real Dataset Challenge (⭐⭐⭐)
- Load a real dataset from sklearn or Kaggle
- Preprocess it appropriately
- Build, train, and evaluate a neural network
- Compare with traditional ML (Random Forest, SVM)
- Which works better? Why?
""")


# BONUS CHALLENGE TEMPLATE
print("\n--- Bonus Challenge Template ---")

# BONUS 1: Early Stopping
def bonus_early_stopping():
    """Add early stopping to prevent overfitting."""
    # TODO: Implement
    pass

# BONUS 2: Learning Rate Experiments
def bonus_learning_rates():
    """Compare different learning rates."""
    # TODO: Implement
    pass

# BONUS 3: Regularization
def bonus_regularization():
    """Add dropout and L2 regularization."""
    from tensorflow.keras import regularizers

    model_regularized = Sequential([
        # TODO: Add layers with regularization
        # Example: layers.Dense(32, activation='relu',
        #                       kernel_regularizer=regularizers.l2(0.01))
        # Example: layers.Dropout(0.5)
    ])
    # TODO: Train and compare
    pass

# BONUS 8: Visualization
def bonus_visualize_layers():
    """Visualize what each layer learns."""
    # TODO: Create intermediate models and visualize activations
    pass


print("\n" + "=" * 60)
print("🎓 KEY TAKEAWAYS")
print("=" * 60)
print("""
1. BINARY vs MULTI-CLASS:
   Binary:      sigmoid activation, binary_crossentropy loss
   Multi-class: softmax activation, categorical_crossentropy loss

2. COMMON MISTAKES:
   ❌ Using sigmoid for multi-class (should be softmax)
   ❌ Using binary_crossentropy for multi-class
   ❌ Forgetting to one-hot encode labels for multi-class
   ❌ Not scaling features

3. DEBUGGING CHECKLIST:
   - Check input shape matches first layer
   - Check output shape matches number of classes
   - Check activation function (sigmoid vs softmax)
   - Check loss function matches problem type
   - Check if labels are properly encoded

4. WORKFLOW:
   Load Data → Scale Features → Build Model → Compile → Train → Evaluate

5. WHEN IN DOUBT:
   - Start simple (1-2 hidden layers)
   - Use 'adam' optimizer
   - Use 'relu' for hidden layers
   - Scale your data!
""")

print("\n✅ Exercise complete! Try the bonus challenges for deeper learning.")
