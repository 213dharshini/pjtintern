# coding: latin-1
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.segmentation import watershed as skwater


# show image
def ShowImage(title, img, ctype):
    plt.figure(figsize=(10, 10))
    if ctype == 'bgr':
        b, g, r = cv2.split(img)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        plt.imshow(rgb_img)
    elif ctype == 'hsv':
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        plt.imshow(rgb)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
    elif ctype == 'rgb':
        plt.imshow(img)
    else:
        raise Exception("Unknown colour type")
    plt.axis('off')
    plt.title(title)
    plt.show()


img = cv2.imread(r'D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/train/train/15.jpg', 0)
img = cv2.medianBlur(img, 5)

ret, thresh = cv2.threshold(img, 100, 250, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 5, 2)

ShowImage('Thresholding Binary image', thresh, 'gray')
ShowImage('Adaptive Thresholding image', thresh2, 'gray')

# watershed algorithm
ret, markers = cv2.connectedComponents(thresh)

# Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
# Get label of largest component by area
largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
# Get pixels which correspond to the brain
retina_mask = markers == largest_component

retina_out = img.copy()
##In a copy of the original image, clear those pixels that don't correspond to the retina
# retina_out[retina_mask==False] = (0,0,0)

img = cv2.imread(r'D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/train/train/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

im1 = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
ShowImage('Watershed segmented image', im1, 'gray')

# canny edge detection

import cv2
import numpy as np

img = cv2.imread(r'D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/train/train/1.jpg', 0)
edges = cv2.Canny(img, 50, 200)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

# CHAN VESE SEGMANTATION
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese

image = gray
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()

# import for tkinter
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

window = tk.Tk()

# Code to add widgets will go here...


window.title("LEAF DISEASE DETECTION")

window.geometry("1200x800")
window.configure(background="yellow")

title = tk.Label(text="CLICK BELOW TO CHOOSE PICTURE FOR TESTING DISEASE....", background="YELLOW", fg="RED",
                 font=("Courier New", 15))
title.grid()

img = ImageTk.PhotoImage(Image.open("org.jpg"))
imglabel = tk.Label(image=img).grid(row=1, column=0)
title = tk.Label(text="HEALTHY LEAF", background="black", fg="yellow", font=("", 12))
title.grid()


def bactDescription():
    top = tk.Tk()
    top.title("DESCRIPTION OF BACTERIAL")
    top.geometry("1000x310")
    top.configure(background="pink")
    title = tk.Label(top, text="DESCRIPTION OF BACTERIAL \n \n \n \n", background="green", fg="white",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Plant pathogenic bacteria cause many different kinds of symptoms that include galls and overgrowths, \n wilts, leaf spots, specks and blights, soft rots,\n as well as scabs and cankers. In contrast to viruses",
                     background="pink", fg="Black", font=("", 15)).place(x=30, y=50)
    title.pack()
 #   uname.pack()
    top.mainloop()


def bactDeficiencies():
    top = tk.Tk()
    top.title("DEFICIENCY OF BACTERIAL")
    top.geometry("1000x310")
    top.configure(background="lightgreen")
    title = tk.Label(top, text="DEFICIENCY OF BACTERIAL \n \n \n \n", background="green", fg="white",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Xylella fastidiosa is a plant pathogenic bacterium that lives inside the host ... Deficiencies in P concentrations in infected leaves were evident",
                     background="pink", fg="red", font=("", 15)).place(x=30, y=50)
    title.pack()
#    uname.pack()
    top.mainloop()


def bactSymptoms():
    top = tk.Tk()
    top.title("SYMPTOMS OF BACTERIAL")
    top.geometry("1000x310")
    top.configure(background="lightblue")
    title = tk.Label(top, text="SYMPTOMS OF BACTERIAL \n \n \n \n", background="blue", fg="white",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Symptoms of bacterial infection in plants are much like \n the symptoms in fungal plant disease.\n They include leaf spots, blights, wilts, scabs, cankers and \n soft rots of roots, storage organs and fruit, and overgrowth.",
                     background="lightblue", fg="red", font=("", 15)).place(x=30, y=50)
    title.pack()
#    uname.pack()
    top.mainloop()


def bactFavourable():
    top = tk.Tk()
    top.title("FAVOUR OF BACTERIAL")
    top.geometry("1000x310")
    top.configure(background="lightblue")
    title = tk.Label(top, text="FAVOURABLE OF BACTERIAL \n \n \n \n", background="blue", fg="white",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Infectious diseases are a significant cause of morbidity and mortality worldwide,\n accounting for approximately 50% of all \n deaths in tropical countries and as much as 20% of deaths in the Americas.\n Despite the significant progress made \n in microbiology and the control of microorganisms",
                     background="lightblue", fg="red", font=("", 15)).place(x=30, y=50)
    title.pack()
#    uname.pack()
    top.mainloop()


def bact():
    window.destroy()
    window1 = tk.Tk()

    window1.title("LEAF DISEASE DETECTION")

    window1.geometry("500x510")
    window1.configure(background="yellow")

    def exit():
        window1.destroy()

    rem = "The remedies for Bacterial Spot are:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                        fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Discard or destroy any affected plants. \n  Do not compost them. \n  Rotate yoour tomato plants yearly to prevent re-infection next year. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def viralDescription():
    top = tk.Tk()
    top.title("DESCRIPTION OF PLANT VIRUS")
    top.geometry("1000x310")
    top.configure(background="brown")
    title = tk.Label(top, text="DESCRIPTION OF PLANT VIRUS \n \n \n \n", background="yellow", fg="red",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Plant viruses are viruses that affect plants.\n Like all other viruses, plant viruses are obligate intracellular parasites that do not have the molecular machinery to replicate without a host.\n Plant viruses can be pathogenic to higher plants.",
                     background="pink", fg="red", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def viralSymptoms():
    top = tk.Tk()
    top.title("SYMPTOMS OF PLANT VIRUS")
    top.geometry("1000x310")
    top.configure(background="blue")
    title = tk.Label(top, text="SYMPTOMS OF PLANT VIRUS \n \n \n \n", background="blue", fg="white",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Infected plants may show a range of symptoms depending on the disease \n but often there is leaf yellowing (either of the whole leaf or in a pattern of stripes or blotches), leaf distortion \n (e.g. curling) and/or other growth distortions in flower or fruit formation).",
                     background="pink", fg="red", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def viralDeficiencies():
    top = tk.Tk()
    top.title("DEFICIENCY OF PLANT VIRUS")
    top.geometry("1000x310")
    top.configure(background="pink")
    title = tk.Label(top, text="DEFICIENCY OF PLANT VIRUS \n \n \n \n", background="yellow", fg="blue",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Some plant diseases are classified as “abiotic,” or \n diseases that are non-infectious and include damage from air pollution, nutritional deficiencies",
                     background="green", fg="white", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def viralFavourable():
    top = tk.Tk()
    top.title("FAVOURABLE CONDITION OF PLANT VIRUS")
    top.geometry("1000x310")
    top.configure(background="lightgreen")
    title = tk.Label(top, text="FAVOURABLE CONDITION OF PLANT VIRUS \n \n \n \n", background="lightblue", fg="blue",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="For a long time green plants, fungi, bacteria and viruses were all treated as plants. ... \n rapidly and therefore can rapidly infect or colonize a favourable habitat.",
                     background="yellow", fg="red", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def vir():
    window.destroy()
    window1 = tk.Tk()

    window1.title("LEAF DISEASE DETECTION")

    window1.geometry("650x510")
    window1.configure(background="pink")

    def exit():
        window1.destroy()

    rem = "The remedies for Yellow leaf curl virus are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                        fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def lateDescription():
    top = tk.Tk()
    top.title("Bacteria Description")
    top.geometry("1000x310")
    top.configure(background="lightblue")
    title = tk.Label(top, text="DESCRIPTION OF LATEBLIGHT \n \n \n \n", background="lightgreen", fg="red",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Plant pathogenic bacteria cause many different kinds of symptoms that include galls and overgrowths, \n wilts, leaf spots, specks and blights, soft rots,\n as well as scabs and cankers. In contrast to viruses,",
                     background="yellow", fg="Black", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def lateSymptoms():
    top = tk.Tk()
    top.title("SYMPTOMS OF LATEBLIGHT")
    top.geometry("980x310")
    top.configure(background="lightgreen")
    title = tk.Label(top, text="SYMPTOMS OF LATEBLIGHT \n \n \n \n", background="lightgreen", fg="red",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="Firm, dark brown, circular spots grow to cover large parts of fruits.\n Spots may become mushy as secondary bacteria invade.\n In high humidity, thin powdery white fungal growth appears on infected leaves, fruit and stems ",
                     background="yellow", fg="black", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def lateDeficiencies():
    top = tk.Tk()
    top.title("DEFICIENCIES OF LATEBLIGHT")
    top.geometry("1000x310")
    top.configure(background="pink")
    title = tk.Label(top, text="DEFICIENCIES OF LATEBLIGHT \n \n \n \n", background="pink", fg="blue",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text=" Late blight is a disease caused by a fungus-like microorganism that infects and  kills tomato and potato plants. \nThe pathogen, Phytophthora infestans, \n was responsible for the Irish potato famine of the 1840's.",
                     background="yellow", fg="black", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def lateFavourable():
    top = tk.Tk()
    top.title("FAVOURABLE OF LATEBLIGHT")
    top.geometry("1000x310")
    top.configure(background="gray")
    title = tk.Label(top, text="FAVOURABLE OF LATEBLIGHT \n \n \n \n", background="green", fg="WHITE",
                     font=("Helvetica", 20))
    uname = tk.Label(top,
                     text="The late blight pathogen is favored by free moisture and cool to moderate temperatures. Night temperatures \n of 50 to 60 F and day temperatures of 60 to 70 F are most favorable for disease development. .",
                     background="lightgreen", fg="Black", font=("", 15)).place(x=30, y=50)
    title.pack()
    uname.pack()
    top.mainloop()


def latebl():
    window.destroy()
    window1 = tk.Tk()

    window1.title("LEAF DISEASE DETECTION")

    window1.geometry("520x510")
    window1.configure(background="pink")

    def exit():
        window1.destroy()

    rem = "The remedies for Late Blight are: "
    remedies = tk.Label(text=rem, background="pink",
                        fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " Monitor the field, remove and destroy infected leaves. \n  Treat organically with copper spray. \n  Use chemical fungicides,the best of which for tomatoes is chlorothalonil."
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm


def analysis():
    # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    verify_data = np.load('verify_data1.npy',allow_pickle=True)
    '''def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()'''

    # using dnn algorithm
    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf


    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    # condition apply

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'viral'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'
        elif np.argmax(model_out) == 4:
            str_label = 'septoria'

        if str_label == 'healthy':
            status = "HEALTHY"
        else:
            status = "UNHEALTHY"

        message = tk.Label(text='STATUS: ' + status, background="pink",
                           fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)
        if str_label == 'bacterial':
            diseasename = "Bacterial Spot "
            disease = tk.Label(text='DISEASE NAME ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button33 = tk.Button(text="REMEDIES", command=bact)
            button33.grid(column=1, row=8, padx=10, pady=10)

            button3 = tk.Button(text="DESCRIPTION", command=bactDescription)
            button3.grid(column=1, row=3, padx=10, pady=10)

            button4 = tk.Button(text="SYMPTOMS", command=bactSymptoms)
            button4.grid(column=1, row=4, padx=10, pady=10)

            button5 = tk.Button(text="DEFICIENCIES", command=bactDeficiencies)
            button5.grid(column=1, row=5, padx=10, pady=10)

            button6 = tk.Button(text="FAVOURABLE CONDITION", command=bactFavourable)
            button6.grid(column=1, row=6, padx=10, pady=10)



        elif str_label == 'viral':
            diseasename = "Yellow leaf curl virus "
            disease = tk.Label(text='DISEASE NAME: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button33 = tk.Button(text="REMEDIES", command=vir)
            button33.grid(column=1, row=8, padx=10, pady=10)

            button3 = tk.Button(text="DESCRIPTION", command=viralDescription)
            button3.grid(column=1, row=3, padx=10, pady=10)

            button4 = tk.Button(text="SYMPTOMS", command=viralSymptoms)
            button4.grid(column=1, row=4, padx=10, pady=10)

            button5 = tk.Button(text="DEFICIENCIES", command=viralDeficiencies)
            button5.grid(column=1, row=5, padx=10, pady=10)

            button6 = tk.Button(text="FAVOURABLE CONDITION", command=viralFavourable)
            button6.grid(column=1, row=6, padx=10, pady=10)

        elif str_label == 'lateblight':
            diseasename = "Late Blight "
            disease = tk.Label(text='DISEASE NAME: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button33 = tk.Button(text="REMEDIES", command=latebl)
            button33.grid(column=1, row=8, padx=10, pady=10)

            button3 = tk.Button(text="DESCRIPTION", command=lateDescription)
            button3.grid(column=1, row=3, padx=10, pady=10)

            button4 = tk.Button(text="SYMPTOMS", command=lateSymptoms)
            button4.grid(column=1, row=4, padx=10, pady=10)

            button5 = tk.Button(text="DEFICIENCIES", command=lateDeficiencies)
            button5.grid(column=1, row=5, padx=10, pady=10)

            button6 = tk.Button(text="FAVOURABLE CONDITION", command=lateFavourable)
            button6.grid(column=1, row=6, padx=10, pady=10)




        elif str_label == 'septoria':
            diseasename = "Tomato Septoria "
            disease = tk.Label(text='DISEASE NAME: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button33 = tk.Button(text="REMEDIES", command=latebl)
            button33.grid(column=1, row=8, padx=10, pady=10)



        else:
            r = tk.Label(text='Plant is healthy', background="lightgreen", fg="Black",
                         font=("", 15))
            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)


# open the image
def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # location of the image which you want to test..... you can change it according to the image location you have
    fileName = askopenfilename(
        initialdir=r'D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/test/test',
        title='Select image for analysis ',
        filetypes=[('image files', '.jpg')])
    dst = r"D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/testpicture"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="250")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=1, row=1, padx=10, pady=10)

    button1.destroy()

    button2 = tk.Button(text="ANALYSE IMAGE", command=analysis)
    button2.grid(column=1, row=2, padx=10, pady=10)


button1 = tk.Button(text="SELECT DISEASE LEAF IMAGE", command=openphoto)
button1.grid(column=1, row=2, padx=10, pady=10)

window.mainloop()





