"""
Support Ticket Classification Model
Portfolio Project: NLP-based Ticket Category Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
csv_path = Path(__file__).parent / "customer_support_tickets_cleaned.csv"

# Validate file exists
if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    raise

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Display basic info
print(df.head(2))
print("\n" + "="*70)

# ============================================================================
# 2. PREPARE FEATURES AND LABELS
# ============================================================================
# Assuming you want to classify by 'Ticket Type' or 'Resolution Speed'
# Modify the 'target_column' based on your CSV structure

target_column = 'Ticket Type'  # Change this to your target column
feature_column = 'Ticket Description'  # Change this to your text column

# Validate required columns exist
if feature_column not in df.columns:
    raise ValueError(f"Feature column '{feature_column}' not found in CSV. Available columns: {df.columns.tolist()}")
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {df.columns.tolist()}")

# Remove rows with missing values in key columns
df_clean = df[[feature_column, target_column]].dropna()

X = df_clean[feature_column].values
y = df_clean[target_column].values

# Ensure X contains string data for text vectorization
if not all(isinstance(val, str) for val in X if pd.notna(val)):
    print("⚠️  Warning: Feature column contains non-string values. Converting to strings...")
    X = np.array([str(val) for val in X])

print(f"Total samples: {len(df_clean)}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution:\n{pd.Series(y).value_counts()}\n")

# Validate minimum samples
if len(df_clean) < 20:
    raise ValueError(f"Not enough samples ({len(df_clean)}). Need at least 20 samples for train/test split.")

# Check if any class has fewer than 2 samples (required for stratified split)
class_counts = pd.Series(y).value_counts()
if (class_counts < 2).any():
    print("⚠️  Warning: Some classes have fewer than 2 samples. Stratified split may fail.")
    print(f"Minimum samples per class: {class_counts.min()}")

# ============================================================================
# 3. SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}\n")

# ============================================================================
# 4. TEXT VECTORIZATION (Convert text to numbers)
# ============================================================================
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(
    max_features=500,      # Use top 500 features
    min_df=2,              # Ignore terms appearing in < 2 documents
    max_df=0.8,            # Ignore terms appearing in > 80% of documents
    ngram_range=(1, 2),    # Use unigrams and bigrams
    stop_words='english'   # Remove English stop words
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_vec.shape}\n")

# ============================================================================
# 5. TRAIN THE MODEL
# ============================================================================
print("Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Handle imbalanced classes
)

model.fit(X_train_vec, y_train)
print("Model training complete!\n")

# ============================================================================
# 6. MAKE PREDICTIONS
# ============================================================================
y_pred = model.predict(X_test_vec)

# ============================================================================
# 7. EVALUATE MODEL PERFORMANCE
# ============================================================================
print("="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

# ============================================================================
# 8. DISPLAY CONFUSION MATRIX
# ============================================================================
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

# ============================================================================
# 9. VISUALIZE RESULTS
# ============================================================================
# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Ticket Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'confusion_matrix.png', dpi=100)
print("✓ Confusion matrix saved as 'confusion_matrix.png'\n")

# ============================================================================
# 10. TEST WITH SAMPLE PREDICTIONS
# ============================================================================
print("="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

sample_texts = X_test[:3]
sample_predictions = model.predict(vectorizer.transform(sample_texts))

for i, text in enumerate(sample_texts):
    print(f"\nTicket {i+1}:")
    print(f"Text: {text[:100]}...")
    print(f"Predicted Class: {sample_predictions[i]}")
    print(f"Actual Class: {y_test[i]}")

print("\n" + "="*70)
print("Model training complete! Ready for deployment.")
print("="*70)
