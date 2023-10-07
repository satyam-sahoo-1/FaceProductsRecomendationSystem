import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Generate a synthetic dataset for demonstration purposes.
# In a real-world scenario, you would need a labeled dataset.
# Here, we create synthetic data with random noise.

# Generate synthetic images with pimples (label 1)
num_samples_with_pimples = 100
synthetic_images_with_pimples = []
# gg = 0
for _ in range(num_samples_with_pimples):
    # Create a synthetic image with a pimple (random noise)
    pimple_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    synthetic_images_with_pimples.append(pimple_image)
#     gg = pimple_image

# cv2.imshow("img",gg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Generate synthetic images without pimples (label 0)
num_samples_without_pimples = 100
synthetic_images_without_pimples = []

for _ in range(num_samples_without_pimples):
    # Create a synthetic image without a pimple (random noise)
    no_pimple_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    synthetic_images_without_pimples.append(no_pimple_image)
#     gg = no_pimple_image

# cv2.imshow("img",gg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Combine the two datasets and create labels
X = np.vstack((synthetic_images_with_pimples, synthetic_images_without_pimples))
y = np.hstack((np.ones(num_samples_with_pimples), np.zeros(num_samples_without_pimples)))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract simple color features (mean color values) for each image
def extract_features(images):
    features = []
    for image in images:
        mean_color = np.mean(image, axis=(0, 1))
        features.append(mean_color)
    return np.array(features)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_features, y_train)

model_filename = 'pimple_detection_model.pkl'
joblib.dump(clf, model_filename)

# Predict on the test set
y_pred = clf.predict(X_test_features)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Pimple detection on a new image (assuming you have a new face image)
new_image = cv2.imread('rohan.jpg')
new_image_features = extract_features([new_image])
prediction = clf.predict(new_image_features)
if prediction == 1:
    print("The image contains a pimple.")
else:
    print("The image does not contain a pimple.")
