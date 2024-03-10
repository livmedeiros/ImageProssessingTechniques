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

### IMAGE INVERSION TECHNIQUE ###

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title('Original image')

inverted_image = 255 - image_np

plt.subplot(1, 2, 2)
plt.imshow(inverted_image)
plt.title('Inverted image')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.hist(image_np.ravel(), bins=256, range=[0, 256])
ax2.hist(inverted_image.ravel(), bins=256, range=[0, 256])
ax1.set_title('Original histogram')
ax2.set_title('Inverted image histogram')

plt.show()