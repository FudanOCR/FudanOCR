from PIL import Image
import numpy as np


def pil_load_img(path):
    image = Image.open(path)

    # Expand the dim if gray
    if image.mode is not 'RGB':
        image = image.convert('RGB')

    image = np.array(image)
    return image
