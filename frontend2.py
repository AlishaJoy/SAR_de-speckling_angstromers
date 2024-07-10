'''
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button, Frame, LEFT
from PIL import Image, ImageTk
import numpy as np
import cv2

def localMean(Im, n):
    kernel = np.ones((2*n+1, 2*n+1)) / ((2*n+1)**2)
    return cv2.filter2D(Im, -1, kernel)

def localVariance(Im, E, n):
    kernel = np.ones((2*n+1, 2*n+1)) / ((2*n+1)**2)
    return cv2.filter2D(Im**2, -1, kernel) - E**2

# Mock implementations for filtering functions
def frost_filter(Im, n=3, a=1.5):
    dim = Im.shape
    deadpixel = 0
    
    E = localMean(Im, n)
    V = localVariance(Im, E, n)
    
    epsilon = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        Speckle_index = V / (E**2 + epsilon)
    
    A_sum = np.zeros(dim)
    weight = np.zeros(dim)

    for k in range(-n, n+1):
        if k < 0:
            Ah = np.pad(Im[:, :dim[1]-abs(k)], ((0, 0), (abs(k), 0)), mode='constant')
        elif k > 0:
            Ah = np.pad(Im[:, k:dim[1]], ((0, 0), (0, k)), mode='constant')
        else:
            Ah = Im

        for i in range(-n, n+1):
            if i < 0:
                Av = np.pad(Ah[:dim[0]-abs(i), :], ((abs(i), 0), (0, 0)), mode='constant')
            elif i > 0:
                Av = np.pad(Ah[i:dim[0], :], ((0, i), (0, 0)), mode='constant')
            else:
                Av = Ah

            dist = np.sqrt(k**2 + i**2)
            factor = np.exp(-Speckle_index * a * dist)
            A_sum += Av * factor
            weight += factor

    Im_filtered = A_sum / weight

    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel

    return Im_filtered

def kaun_filter(Im, n=3, sigma=0.5):
    dim = Im.shape
    deadpixel = 0

    E = localMean(Im, n)
    V = localVariance(Im, E, n)
    
    epsilon = 1e-10
    C = V / (E**2 + epsilon)
    Cu = sigma**2 / (E**2 + epsilon)
    
    W = Cu / (Cu + C)
    
    Im_filtered = E + W * (Im - E)
    
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel
    
    return Im_filtered

def lee_filter(Im, n=3, sigma=0.5):
    dim = Im.shape
    deadpixel = 0
    
    E = localMean(Im, n)
    V = localVariance(Im, E, n)

    epsilon = 1e-10
    Cu = sigma**2 / (E**2 + epsilon)
    
    W = Cu / (Cu + V / (E**2 + epsilon))
    
    Im_filtered = E + W * (Im - E)
    
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel
    
    return Im_filtered

class SARImageDespecklingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Image Despeckling")
        
        self.input_image = None
        # Heading
        self.heading = Label(root, text="SAR Image Despeckling", font=("Helvetica", 16))
        self.heading.pack()
         # Buttons and Frames
        self.upload_button = Button(root, text="Upload", command=self.upload_image)
        self.upload_button.grid(row=1, column=0, padx=5, pady=5)
        self.frost_button = Button(root, text="Frost Filter", command=self.apply_frost_filter)
        self.frost_button.grid(row=1, column=1, padx=5, pady=5)
        self.lee_button = Button(root, text="Lee Filter", command=self.apply_lee_filter)
        self.lee_button.grid(row=1, column=2, padx=5, pady=5)
        self.kaun_button = Button(root, text="Kaun Filter", command=self.apply_kaun_filter)
        self.kaun_button.grid(row=1, column=3, padx=5, pady=5)
        
        self.input_frame = Frame(root, width=200, height=200, bg="gray")
        self.input_frame.grid(row=2, column=0, padx=10, pady=10)
        self.output_frame1 = Frame(root, width=200, height=200, bg="gray")
        self.output_frame1.grid(row=2, column=1, padx=10, pady=10)
        self.output_frame2 = Frame(root, width=200, height=200, bg="gray")
        self.output_frame2.grid(row=2, column=2, padx=10, pady=10)
        self.output_frame3 = Frame(root, width=200, height=200, bg="gray")
        self.output_frame3.grid(row=2, column=3, padx=10, pady=10)
        ##########
        Buttons
        self.upload_button = Button(root, text="Upload", command=self.upload_image)
        self.upload_button.pack(side=LEFT, padx=5, pady=5)
        self.frost_button = Button(root, text="Frost Filter", command=self.apply_frost_filter)
        self.frost_button.pack(side=LEFT, padx=5, pady=5)
        self.lee_button = Button(root, text="Lee Filter", command=self.apply_lee_filter)
        self.lee_button.pack(side=LEFT, padx=5, pady=5)
        self.kaun_button = Button(root, text="Kaun Filter", command=self.apply_kaun_filter)
        self.kaun_button.pack(side=LEFT, padx=5, pady=5) # Frames
        self.input_frame = Frame(root, width=200, height=200, bg="gray")
        self.input_frame.pack(side=LEFT, padx=10, pady=10)
        self.output_frame1 = Frame(root, width=200, height=200, bg="gray")
        self.output_frame1.pack(side=LEFT, padx=10, pady=10)
        self.output_frame2 = Frame(root, width=200, height=200, bg="gray")
        self.output_frame2.pack(side=LEFT, padx=10, pady=10)
        self.output_frame3 = Frame(root, width=200, height=200, bg="gray")
        self.output_frame3.pack(side=LEFT, padx=10, pady=10)
        ############
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.input_image, self.input_frame)
        
    def display_image(self, image, frame):
        for widget in frame.winfo_children():
            widget.destroy()
        img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=img)
        label = Label(frame, image=img_tk)
        label.image = img_tk
        label.pack()
        
    def apply_frost_filter(self):
        if self.input_image is not None:
            filtered_image = frost_filter(self.input_image)
            self.display_image(filtered_image, self.output_frame1)
        
    def apply_lee_filter(self):
        if self.input_image is not None:
            filtered_image = lee_filter(self.input_image)
            self.display_image(filtered_image, self.output_frame2)
        
    def apply_kaun_filter(self):
        if self.input_image is not None:
            filtered_image = kaun_filter(self.input_image)
            self.display_image(filtered_image, self.output_frame3)

if __name__ == "__main__":
    root = tk.Tk()
    app = SARImageDespecklingApp(root)
    root.mainloop()'''


#########################
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import random
import cv2

# Helper functions for local mean and variance
def localMean(Im, n):
    kernel = np.ones((2*n+1, 2*n+1)) / ((2*n+1)**2)
    return cv2.filter2D(Im, -1, kernel)

def localVariance(Im, E, n):
    kernel = np.ones((2*n+1, 2*n+1)) / ((2*n+1)**2)
    return cv2.filter2D(Im**2, -1, kernel) - E**2

# Filtering functions
def frost_filter(Im, n=3, a=1.5):
    dim = Im.shape
    deadpixel = 0
    
    E = localMean(Im, n)
    V = localVariance(Im, E, n)
    
    epsilon = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        Speckle_index = V / (E**2 + epsilon)
    
    A_sum = np.zeros(dim)
    weight = np.zeros(dim)

    for k in range(-n, n+1):
        if k < 0:
            Ah = np.pad(Im[:, :dim[1]-abs(k)], ((0, 0), (abs(k), 0)), mode='constant')
        elif k > 0:
            Ah = np.pad(Im[:, k:dim[1]], ((0, 0), (0, k)), mode='constant')
        else:
            Ah = Im

        for i in range(-n, n+1):
            if i < 0:
                Av = np.pad(Ah[:dim[0]-abs(i), :], ((abs(i), 0), (0, 0)), mode='constant')
            elif i > 0:
                Av = np.pad(Ah[i:dim[0], :], ((0, i), (0, 0)), mode='constant')
            else:
                Av = Ah

            dist = np.sqrt(k**2 + i**2)
            factor = np.exp(-Speckle_index * a * dist)
            A_sum += Av * factor
            weight += factor

    Im_filtered = A_sum / weight

    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel

    return Im_filtered

def kuan_filter(Im, n=3, sigma=0.5):
    dim = Im.shape
    deadpixel = 0

    E = localMean(Im, n)
    V = localVariance(Im, E, n)
    
    epsilon = 1e-10
    C = V / (E**2 + epsilon)
    Cu = sigma**2 / (E**2 + epsilon)
    
    W = Cu / (Cu + C)
    
    Im_filtered = E + W * (Im - E)
    
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel
    
    return Im_filtered

def lee_filter(Im, n=3, sigma=0.5):
    dim = Im.shape
    deadpixel = 0
    
    E = localMean(Im, n)
    V = localVariance(Im, E, n)

    epsilon = 1e-10
    Cu = sigma**2 / (E**2 + epsilon)
    
    W = Cu / (Cu + V / (E**2 + epsilon))
    
    Im_filtered = E + W * (Im - E)
    
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel
    
    return Im_filtered

class SARImageDespecklingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Image Despeckling")
        
        self.input_image = None
        
        # Heading
        self.heading = Label(root, text="SAR Image Despeckling", font=("Helvetica", 16))
        self.heading.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Buttons and Frames
        self.upload_button = Button(root, text="Upload", command=self.upload_image)
        self.upload_button.grid(row=1, column=0, padx=5, pady=5)
        self.frost_button = Button(root, text="Frost Filter", command=self.apply_frost_filter)
        self.frost_button.grid(row=1, column=1, padx=5, pady=5)
        self.lee_button = Button(root, text="Lee Filter", command=self.apply_lee_filter)
        self.lee_button.grid(row=1, column=2, padx=5, pady=5)
        self.kaun_button = Button(root, text="Kaun Filter", command=self.apply_kaun_filter)
        self.kaun_button.grid(row=1, column=3, padx=5, pady=5)
        
        self.input_frame = Frame(root, width=200, height=200, bg="pink")
        self.input_frame.grid(row=2, column=0, padx=10, pady=10)
        self.output_frame1 = Frame(root, width=200, height=200, bg="pink")
        self.output_frame1.grid(row=2, column=1, padx=10, pady=10)
        self.output_frame2 = Frame(root, width=200, height=200, bg="pink")
        self.output_frame2.grid(row=2, column=2, padx=10, pady=10)
        self.output_frame3 = Frame(root, width=200, height=200, bg="pink")
        self.output_frame3.grid(row=2, column=3, padx=10, pady=10)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.input_image, self.input_frame)
        
    def display_image(self, image, frame):
        for widget in frame.winfo_children():
            widget.destroy()
        img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=img)
        label = Label(frame, image=img_tk)
        label.image = img_tk
        label.pack()
        
    def apply_frost_filter(self):
        if self.input_image is not None:
            filtered_image = frost_filter(self.input_image)
            self.display_image(filtered_image, self.output_frame1)
        
    def apply_lee_filter(self):
        if self.input_image is not None:
            filtered_image = lee_filter(self.input_image)
            self.display_image(filtered_image, self.output_frame2)
        
    def apply_kaun_filter(self):
        if self.input_image is not None:
            filtered_image = kuan_filter(self.input_image)
            self.display_image(filtered_image, self.output_frame3)

if __name__ == "__main__":
    root = tk.Tk()
    app = SARImageDespecklingApp(root)
    root.mainloop()
