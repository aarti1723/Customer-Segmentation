import pandas as pd
from sklearn.cluster import KMeans
import pickle
import os

# Load valid CSV data
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("✅ Model saved at model/kmeans_model.pkl")