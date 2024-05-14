import tkinter as tk
import cv2
import numpy as np
from tkinter import Scale, Button, Label, Tk
from PIL import Image, ImageTk
import os

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Page 2")
        self.slider_row = 2
        
        self.image_label = Label(master)
        self.image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")  # Image label at top
        
        self.load_image_button = Button(self.master, text="Load image",bg="Purple",fg="white",width=10,height=1,font=30,activebackground="red",cursor="bottom_side", command=self.load_image)
        self.load_image_button.grid(row=1, column=0, pady=10)  
        self.Next_Page_button = Button(self.master, text="Next Page",bg="green",fg="white",width=12,height=1,font=30,cursor="sb_right_arrow", command=self.open_page3)
        self.Next_Page_button.grid(row=7, column=1, pady=10) 
        self.Last_Page_button = Button(self.master, text="Previous page",bg="red",fg="white",width=12,height=1,font=30,cursor="sb_left_arrow", command=self.open_page1)
        self.Last_Page_button.grid(row=7, column=0, pady=10 ) 
        self.add_buttons_and_sliders()
        self.load_default_image()

    def load_default_image(self):
        path = "image.jpg"
        self.original_image = cv2.imread(path)
        self.update_image(self.original_image)

    def load_image(self):
        path = "image.jpg"
        self.original_image = cv2.imread(path)
        self.update_image(self.original_image)
        

    def update_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Resize image to fit within a maximum size
        max_width = 800
        max_height = 500
        image.thumbnail((max_width, max_height))
        
        image = ImageTk.PhotoImage(image)

        self.image_label.configure(image=image)
        self.image_label.image = image

    def add_buttons_and_sliders(self):
        self.add_button("Roberts_ED", self.apply_roberts_edge_detector,1)

        self.add_button("Prewitt_ED", self.apply_prewitt_edge_detector,2)

        self.add_button("Sobel_ED", self.apply_sobel_edge_detector,3)


        

    def add_button(self, text, command,row):
        button = Button(self.master, text=text, command=command,bg="blue",fg="white",width=20,height=1,font=("verdana",13,"bold"),cursor="hand2",activebackground="Green")
        button.grid(row=(row+1), column=0,pady=10)  # Set the row parameter to position the button

        

    def apply_roberts_edge_detector(self):
        roberts_image = cv2.Canny(self.original_image, 100, 200)
        self.update_image(roberts_image)

    def apply_prewitt_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        prewitt_image = np.sqrt(prewitt_x**2 + prewitt_y**2)
        prewitt_image = cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.update_image(prewitt_image)

    def apply_sobel_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.update_image(sobel_image)

    def open_page3(self):
            os.system("python Page3.py")  

    def open_page1(self):
            os.system("python Page1.py")  

root = Tk()
app = ImageProcessingApp(root)
root.mainloop()
