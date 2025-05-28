
import numpy as np
from PIL import Image
def is_empty_image(image: Image)->bool:
    return np.all(np.asarray(image)==0)
