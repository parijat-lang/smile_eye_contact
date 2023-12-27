import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
facemesh_images = np.load('smile_images.npy')
labels = np.load('smile_labels.npy')

# Reshape and scale data
X = facemesh_images.reshape(facemesh_images.shape[0], -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.001, random_state=42)

# Create and train the Logistic Regression model with increased max_iter and different solver
logistic_model = LogisticRegression(max_iter=3000, solver='saga', random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open('smile_logistic_model.pkl', 'wb') as file:
    pickle.dump(logistic_model, file)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)