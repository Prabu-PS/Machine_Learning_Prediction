# for data manipulation
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

# Authenticate to Hugging Face (needed for hf:// access)
login(token=os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset from Hugging Face
DATASET_PATH = "hf://datasets/PSstark/Machine-Learning-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully from Hugging Face.")

# Drop the unique identifier
df.drop(columns=['UDI'], inplace=True)

# Encode categorical 'Type' column
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

target_col = 'ProdTaken'

# Split data
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="PSstark/Machine-Learning-Prediction",
        repo_type="dataset",
    )

print("✅ Data prep completed and files uploaded to Hugging Face.")
