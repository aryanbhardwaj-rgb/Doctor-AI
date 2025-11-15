import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB  # <-- THE MODEL FIX
import joblib
import json

print("Starting AI Doctor Model Training (Sensible & Robust)...")

try:
    # 1. Load Data
    df = pd.read_csv('dataset.csv')
    print("Loaded dataset.csv")
except FileNotFoundError:
    print("Error: 'dataset.csv' not found.")
    print("Please download it from: https://www.kaggle.com/datasets/itachi9/disease-symptom-prediction")
    exit()

# 2. Preprocess Data
print("Cleaning and preprocessing data...")
cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
        'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
        'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
        'Symptom_16', 'Symptom_17']

df_melted = df.melt(id_vars=['Disease'], value_vars=cols, value_name='Symptom')
df_melted = df_melted.dropna().drop(columns=['variable'])
df_melted['Symptom'] = df_melted['Symptom'].str.strip().str.replace(' ', '_')

data = pd.crosstab(df_melted['Disease'], df_melted['Symptom'])
data = data.reset_index()

# 3. Create Features (X) and Labels (y)
print("Creating features (X) and labels (y)...")
le = LabelEncoder()
y = le.fit_transform(data['Disease'])
X = data.drop(columns=['Disease'])

# Get symptom column names for saving later
symptom_cols = list(X.columns)

# 4. Train the Model
print("Training the AI model (Multinomial Naive Bayes)...")
# Split data to check accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === THE FIX ===
# We are using MultinomialNB because it's superior for this type of
# probabilistic, sparse-data problem. It gives more sensible answers.
model = MultinomialNB()
# ===================

model.fit(X_train, y_train)

# Save the disease names (classes) with the model
model.classes_ = le.classes_

print(f"Model trained. Accuracy on test split: {model.score(X_test, y_test) * 100:.2f}%")

# 5. Save the Final Model
print("Saving model to 'disease_model.pkl'...")
# Train on the *full* dataset for the best possible model
final_model = MultinomialNB()
final_model.fit(X, y) # Train on ALL data
final_model.classes_ = le.classes_ # Don't forget to add classes again!

joblib.dump(final_model, 'disease_model.pkl')

print("Saving symptom list to 'symptom_columns.json'...")
with open('symptom_columns.json', 'w') as f:
    json.dump(symptom_cols, f)

print("\n--- Success! ---")
print("Your new SENSIBLE AI model ('disease_model.pkl') is ready.")
print("You can now run the Streamlit app: streamlit run main.py")