import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Filter functions
def localMean(Im, n):
    kernel = np.ones((2*n+1, 2*n+1), np.float32) / (2*n+1)**2
    return cv2.filter2D(Im, -1, kernel)

def localVariance(Im, E, n):
    kernel = np.ones((2*n+1, 2*n+1), np.float32) / (2*n+1)**2
    E2 = cv2.filter2D(Im**2, -1, kernel)
    return E2 - E**2

def frost_filter(Im, n, a):
    dim = Im.shape
    deadpixel = 0
    
    # Local values
    E = localMean(Im, n)
    V = localVariance(Im, E, n)
    
    epsilon = 1e-10  # Small constant to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        Speckle_index = V / (E**2 + epsilon)
    
    A_sum = np.zeros(dim)
    weight = np.zeros(dim)

    # Frost filter code
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

    # Assign dead pixels
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel

    return Im_filtered

def kuan_filter(Im, n, sigma):
    dim = Im.shape
    deadpixel = 0

    # Local values
    E = localMean(Im, n)
    V = localVariance(Im, E, n)
    
    epsilon = 1e-10  # Small constant to avoid division by zero
    C = V / (E**2 + epsilon)
    Cu = sigma**2 / (E**2 + epsilon)
    
    # Filter weight
    W = Cu / (Cu + C)
    
    # Filtered image
    Im_filtered = E + W * (Im - E)
    
    # Assign dead pixels
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel
    
    return Im_filtered

def lee_filter(Im, n, sigma):
    dim = Im.shape
    deadpixel = 0
    
    # Local values
    E = localMean(Im, n)
    V = localVariance(Im, E, n)

    epsilon = 1e-10  # Small constant to avoid division by zero
    Cu = sigma**2 / (E**2 + epsilon)
    
    # Filter weight
    W = Cu / (Cu + V / (E**2 + epsilon))
    
    # Filtered image
    Im_filtered = E + W * (Im - E)
    
    # Assign dead pixels
    Im_filtered[dim[0]-n:dim[0], :] = deadpixel
    Im_filtered[:n, :] = deadpixel
    Im_filtered[:, dim[1]-n:dim[1]] = deadpixel
    Im_filtered[:, :n] = deadpixel
    
    return Im_filtered

class SARFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Image Filtering")
        self.root.geometry("800x600")
        
        # Frame for file selection
        self.frame1 = ttk.LabelFrame(root, text="File Selection")
        self.frame1.pack(fill="both", expand="yes", padx=10, pady=10)
        
        # Entry to display selected file
        self.filepath = tk.StringVar()
        self.file_entry = ttk.Entry(self.frame1, textvariable=self.filepath, width=50)
        self.file_entry.pack(side="left", padx=10, pady=10)
        
        # Button to select file
        self.browse_button = ttk.Button(self.frame1, text="Browse", command=self.browse_file)
        self.browse_button.pack(side="left", padx=10, pady=10)
        
        # Frame for filter selection
        self.frame2 = ttk.LabelFrame(root, text="Filter Selection")
        self.frame2.pack(fill="both", expand="yes", padx=10, pady=10)
        
        # Combobox to select filter
        self.filter_var = tk.StringVar()
        self.filter_combo = ttk.Combobox(self.frame2, textvariable=self.filter_var, state="readonly")
        self.filter_combo['values'] = ("Frost Filter", "Kuan Filter", "Lee Filter")
        self.filter_combo.current(0)
        self.filter_combo.pack(side="left", padx=10, pady=10)
        
        # Button to apply filter
        self.apply_button = ttk.Button(self.frame2, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack(side="left", padx=10, pady=10)
        
        # Frame for displaying images
        self.frame3 = ttk.LabelFrame(root, text="Image Display")
        self.frame3.pack(fill="both", expand="yes", padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.frame3)
        self.canvas.pack(fill="both", expand="yes")
        
        self.image_on_canvas = None

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")])
        if file_path:
            self.filepath.set(file_path)
            self.display_image(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((400, 400), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image)
        self.canvas.create_image(200, 200, image=self.img)
        
    def apply_filter(self):
        file_path = self.filepath.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an image file")
            return
        
        filter_name = self.filter_var.get()
        Im = Image.open(file_path)
        Im = np.array(Im.convert('L'))  # Convert image to grayscale
        
        if filter_name == "Frost Filter":
            filtered_image = frost_filter(Im, 3, 1)
        elif filter_name == "Kuan Filter":
            filtered_image = kuan_filter(Im, 3, 25)
        elif filter_name == "Lee Filter":
            filtered_image = lee_filter(Im, 3, 25)
        
        self.show_filtered_image(filtered_image)
    
    def show_filtered_image(self, image):
        image = Image.fromarray(np.uint8(image))
        image.thumbnail((400, 400), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image)
        self.canvas.create_image(200, 200, image=self.img)

if __name__ == "__main__":
    root = tk.Tk()
    app = SARFilterApp(root)
    root.mainloop()
