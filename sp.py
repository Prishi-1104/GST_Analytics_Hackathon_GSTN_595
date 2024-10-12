import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and testing datasets
x_train_data = pd.read_csv('data/X_Train_Data_Input.csv')  # Train features
y_train_data = pd.read_csv('data/Y_Train_Data_Target.csv')  # Train target

x_test_data = pd.read_csv('data/X_Test_Data_Input.csv')    # Test features
y_test_data = pd.read_csv('data/Y_Test_Data_Target.csv')    # Test target

# Merge the training datasets on 'ID'
train_data = pd.merge(x_train_data, y_train_data[['ID', 'target']], on='ID')

# Merge the test datasets on 'ID'
test_data = pd.merge(x_test_data, y_test_data[['ID', 'target']], on='ID')

# Prepare the training data (X_train and y_train)
X_train = train_data.drop(['ID', 'target'], axis=1)
y_train = train_data['target']

# Prepare the test data (X_test and y_test)
X_test = test_data.drop(['ID', 'target'], axis=1)
y_test = test_data['target']

# Handle any missing values by replacing them with column mean
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")
print("\nClassification Report:")
print(classification_rep)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Reds', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
