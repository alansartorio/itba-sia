import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()
modeldir = args.model


from tkinter import *
import model
import numpy as np
from PIL import Image, ImageTk
import imageUtils
import os

outputDir = 'outputImages'


currentImage: Image.Image = None

master = Tk()

label = Label(master, image=None)
label.pack(expand='yes')

def draw():
    img = ImageTk.PhotoImage(image=currentImage.resize((500, 500), resample=Image.NEAREST))
    # img = ImageTk.PhotoImage(image=Image.fromarray(np.array((300, 300, 3)).resize((500, 500), resample=Image.NEAREST))
    label.configure(image=img)
    label.image = img

sliders: list[Scale] = []
def getLatent():
    return np.array([slider.get() for slider in sliders])


sliderLimits = 5
def getCellPos():
    l = len(sliders)
    return dict(row=l%10, column=l//10)
slidersPanel = Frame(master)
slidersPanel.pack()
def addSlider():
    w = Scale(slidersPanel, from_=-sliderLimits, to=sliderLimits, orient=HORIZONTAL, command=lambda event:updateImage(), resolution=0.01)
    w.grid(**getCellPos())
    sliders.append(w)

vae = model.VAE.load(modeldir)
if vae is None:
    print("Model does not exist")
    exit(1)

def getImage():
    latent = getLatent()
    img = vae.predict(latent)
    # return Image.fromarray((img * 255).astype(np.uint8), 'RGB')
    return Image.fromarray((img[:,:,0] * 255).astype(np.uint8), 'L')



def updateImage():
    global currentImage
    currentImage = getImage()
    draw()


for _ in range(vae.latentDim):
    addSlider()

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

lastIndex = 0

def getAvailablePath():
    global lastIndex
    def getImagePath():
        return os.path.join(outputDir, f'image{lastIndex}.png')
    while os.path.exists(getImagePath()):
        lastIndex += 1
    return getImagePath()

def saveImage():
    newSize = 36*8
    currentImage.resize((newSize, newSize), resample=Image.NEAREST).save(getAvailablePath())

button = Button(master, text="Save", command=saveImage)
button.pack()

updateImage()

mainloop()
