import argparse
import model
import imageUtils
# from albumentations import Compose, Rotate, HorizontalFlip, JpegCompression
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('images')
parser.add_argument('--override', action='store_const',
                    const=True, default=False)
args = parser.parse_args()
modelsDir = args.model
imagesDir = args.images
override = args.override

# transforms = Compose([
    # Rotate(limit=180),
    # JpegCompression(),
    # HorizontalFlip(),
# ])

# size = (36, 36, 3)

images = imageUtils.loadImages(imagesDir)
print(images.shape)
size = images[0].shape


# showImage(images[0])

# imageUtils.showImage(images[0])
# exit(0)


def loadOrCreate(forceCreate: bool = False):
    vae = None
    if not forceCreate:
        vae = model.VAE.load(modelsDir)
    if vae is None:
        # vae = model.buildModel(size, 10)
        vae = model.buildNonConvModel(size, 5)
    return vae


vae = loadOrCreate(override)
try:
    vae.train(images, epochs=5000, batch_size=2)
except KeyboardInterrupt:
    pass
finally:
    vae.save(modelsDir)

# print(vae.getLatent(images[0]))
imageUtils.showImage(vae.predict(vae.getLatent(images[0])))

