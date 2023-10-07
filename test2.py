import cv2
import numpy as np
# from skimage.feature import hog
import joblib  # Import joblib for model loading

# Extract simple color features (mean color values) for each image
def extract_features(images):
    features = []
    for image in images:
        mean_color = np.mean(image, axis=(0, 1))
        features.append(mean_color)
    return np.array(features)

# Load the trained model from the saved file
model_filename = 'pimple_detection_model.pkl'
clf = joblib.load(model_filename)

# Pimple detection on a new image (assuming you have a new face image)
new_image = cv2.imread('satyam.jpg')
new_image_features = extract_features([new_image])
prediction = clf.predict(new_image_features)
if prediction == 1:
    print("The image contains a pimple.")
else:
    print("The image does not contain a pimple.")
