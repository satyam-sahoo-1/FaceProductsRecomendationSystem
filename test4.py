import cv2
import numpy as np

def detect_skin_tone(image):
  """Detects the skin tone of the person in the image.

  Args:
    image: A numpy array representing the image.

  Returns:
    A string representing the skin tone, which can be "dark", "moderate", or "white".
  """

  # Convert the image to the HSV color space.
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Calculate the average HSV values of the skin.
  skin_hsv = np.mean(hsv, axis=(0, 1))

  # Determine the skin tone based on the average HSV values.
  if skin_hsv[0] < 20 or skin_hsv[2] < 100:
    skin_tone = "dark"
  elif skin_hsv[0] < 60 or skin_hsv[2] < 180:
    skin_tone = "moderate"
  else:
    skin_tone = "white"

  return skin_tone


# Load the image.
image = cv2.imread("satyam.jpg")

# Detect the skin tone.
skin_tone = detect_skin_tone(image)

# Print the skin tone.
print(skin_tone)