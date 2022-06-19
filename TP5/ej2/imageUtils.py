from glob import glob
from os import path
import numpy as np
from PIL import Image

def loadImages(dir: str, size: tuple[int, int, int] = None):
    images = []

    exts = ['jpg', 'png']
    for ext in exts:
        for imageFile in glob(path.join(dir, f'*.{ext}')):
            image = Image.open(imageFile)
            # image = image.convert('RGBA')
            # # print(imageFile, image)
            # image = Image.alpha_composite(Image.new("RGBA", image.size, "WHITE"), image)
            # image = image.convert('RGB')
            image = np.array(image, dtype=np.uint8)
            image = np.expand_dims(image, -1)
            if image.shape[-1] == 4:
                image = image[:,:,:-1]
            if size is None:
                size = image.shape
            if image.shape != size:
                raise Exception(f"Image \"{imageFile}\" does not have the expected size ({size})")
            image = image.astype(np.float32) / 255
            images.append(image)
        
    return np.array(images)

def showImage(image: np.ndarray, scale: bool = True):
    if image.shape[2] == 1:
        image = image[:,:,0]
    Image.fromarray((image * 255).astype(np.uint8)).resize((200, 200), resample=Image.NEAREST).show()
