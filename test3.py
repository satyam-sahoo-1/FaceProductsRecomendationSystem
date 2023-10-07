import cv2
import numpy as np

# Load an image
image = cv2.imread('images.jpg')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for skin colors in HSV
# These values can be adjusted based on your definition of "dark," "moderate," and "white" skin tones
lower_dark_skin = np.array([0, 20, 40], dtype=np.uint8)
upper_dark_skin = np.array([20, 255, 255], dtype=np.uint8)

lower_moderate_skin = np.array([0, 20, 100], dtype=np.uint8)
upper_moderate_skin = np.array([20, 255, 255], dtype=np.uint8)

lower_white_skin = np.array([0, 0, 150], dtype=np.uint8)
upper_white_skin = np.array([20, 30, 255], dtype=np.uint8)

# Create masks for different skin tone categories
dark_skin_mask = cv2.inRange(hsv_image, lower_dark_skin, upper_dark_skin)
moderate_skin_mask = cv2.inRange(hsv_image, lower_moderate_skin, upper_moderate_skin)
white_skin_mask = cv2.inRange(hsv_image, lower_white_skin, upper_white_skin)

# Calculate the area of each skin tone category
dark_skin_area = np.sum(dark_skin_mask > 0)
moderate_skin_area = np.sum(moderate_skin_mask > 0)
white_skin_area = np.sum(white_skin_mask > 0)

# Determine the predominant skin tone category based on area
if dark_skin_area > moderate_skin_area and dark_skin_area > white_skin_area:
    skin_tone_category = "Dark"
elif moderate_skin_area > dark_skin_area and moderate_skin_area > white_skin_area:
    skin_tone_category = "Moderate"
else:
    skin_tone_category = "White"

# Display the skin tone category
print(f"Skin Tone Category: {skin_tone_category}")
