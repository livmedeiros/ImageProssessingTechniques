import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

url = "https://i.stack.imgur.com/TPVnd.png"

# Request to get image from URL
response = requests.get(url)

image = Image.open(BytesIO(response.content))
image_np = np.array(image)

### THRESHOLDING TECHNIQUE ###

# Converting image to grayscale
if len(image_np.shape) > 2:
    gray_image = np.mean(image_np, axis=2).astype(np.uint8)
else:
    gray_image = image_np

threshold_value = 100

binary_image = np.where(gray_image > threshold_value, 255, 0).astype(np.uint8)

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original image (grayscale)')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded image')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.hist(gray_image.ravel(), bins=256, range=[0, 256])
ax2.hist(binary_image.ravel(), bins=256, range=[0, 256])
ax1.set_title('Original histogram')
ax2.set_title('Thresholded image histogram')

plt.show()