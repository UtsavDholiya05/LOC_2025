import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Step 1: Load the dataset
column_names = [
    'status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_account',
    'employment_since', 'installment_rate', 'personal_status', 'other_debtors', 'residence_since',
    'property', 'age', 'other_installment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'telephone', 'foreign_worker', 'risk'
]

data = pd.read_csv('german.data', sep=' ', header=None, names=column_names)

# Step 2: Display basic information about the dataset
print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())

# Step 3: Identify and encode categorical variables
categorical_cols = [
    'status', 'credit_history', 'purpose', 'savings_account', 'employment_since',
    'personal_status', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job',
    'telephone', 'foreign_worker'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 4: Convert target variable to binary (1 = Good, 0 = Bad)
data['risk'] = data['risk'].apply(lambda x: 0 if x == 2 else 1)

# Step 5: Separate features and target variable
X = data.drop(columns=['risk'])  # Features
y = data['risk']  # Target variable

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Scale numerical features
scaler = StandardScaler()
numerical_cols = ['duration', 'credit_amount', 'installment_rate', 'residence_since', 'age', 'existing_credits', 'num_dependents']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Step 8: Train the model with balanced class weights
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Step 11: Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 12: Save the model and preprocessing objects
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)