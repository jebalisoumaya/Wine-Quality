import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Load dataset
df = pd.read_csv('WineQT.csv')

# Display basic information
print(df.head(5))
print(df.info())  # all variables are numerical, no need for conversion
print(df.describe())

# Check for missing values
print(df.isnull().sum())  # No missing values

# Check for duplicates
print(df.duplicated().sum())  # No duplicates

# Number of wines in the dataset
num_wines = df.shape[0]
print(f'Number of wines in the dataset: {num_wines}')

# Calculate Correlation Matrix to identify important features
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# **************** Decision Tree ********************
# From the correlation matrix, select features with strong correlations with 'quality':
# - Alcohol (0.48)
# - Volatile Acidity (-0.41)
# - Citric Acid (0.24)
# - Sulphates (0.26)
# - Fixed Acidity (0.12)

# Define X (input features) and y (target output variable 'quality')
X = df[['alcohol', 'volatile acidity', 'citric acid', 'sulphates', 'fixed acidity']]

# Convert 'quality' into 3 classes: Low (0), Medium (1), High (2)
def categorize_quality(quality):
    if quality <= 4:
        return 0  # Low
    elif quality <= 6:
        return 1  # Medium
    else:
        return 2  # High

y = df['quality'].apply(categorize_quality)
print(y.value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=5)  # Limiting depth and min samples per leaf
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 12))
plot_tree(clf, feature_names=X.columns, class_names=[str(i) for i in clf.classes_],
          filled=True, rounded=True, fontsize=12, proportion=True)
plt.title("Decision Tree Classifier", fontsize=16)
plt.tight_layout()
plt.show()

# Evaluate the model
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report (Precision, Recall, F1-score, etc.)
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
