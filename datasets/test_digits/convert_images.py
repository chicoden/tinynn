from PIL import Image
import numpy as np
from pyperclip import copy
import os

images = "{\n"
for file in os.listdir("."):
    if not file.endswith(".png"):
        continue

    image = np.asarray(np.array(Image.open(file))[:, :, 1], dtype=np.float32) / 255
    image_str = ""
    for row in image:
        image_str += f"        {row[0]:.2f}f"
        for pixel in row[1:]:
            image_str += f", {pixel:.2f}f"
        image_str += ",\n"
    images += "    {\n" + image_str[:-2] + "\n    },\n"
images = images[:-2] + "\n};"

copy(images)
