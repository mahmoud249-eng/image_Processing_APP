import tkinter as tk
import cv2
import numpy as np
from tkinter import Scale, Button, Label, Tk
from PIL import Image, ImageTk
import os

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Page 4")
        self.slider_row = 2
        
        self.image_label = Label(master)
        self.image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")  # Image label at top
        
        self.load_image_button = Button(self.master, text="Load image",bg="Purple",fg="white",width=10,height=1,font=30,activebackground="red",cursor="bottom_side", command=self.load_image)
        self.load_image_button.grid(row=1, column=0, pady=10)  # Load image button below image
        self.Last_Page_button = Button(self.master, text="Previous page",bg="red",fg="white",width=12,height=1,font=30,cursor="sb_left_arrow", command=self.open_page3)
        self.Last_Page_button.grid(row=7, column=1, pady=10) 
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
        self.slider_row = 2
        self.add_slider(1, 250, 0, self.update_thresholding_segmentation)


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
        self.add_button("thresholding_segmentation", self.apply_thresholding_segmentation,1)
        self.add_slider(1, 250, 0, self.update_thresholding_segmentation)
        self.add_button("hough_circle_transform", self.apply_hough_circle_transform,2)
        self.add_button("region_split_merge_seg", self.apply_region_split_merge_segmentation,3)

    def add_button(self, text, command,row):
        button = Button(self.master, text=text, command=command,bg="Coral",fg="white",width=20,height=1,font=("verdana",13,"bold"),cursor="hand2",activebackground="Green")
        button.grid(row=(row+1), column=0, pady=10)  # # Set the row parameter to position the button

    def add_slider(self,from_, to_, default, command):
        slider_label = Label(self.master)
        slider_label.grid(row=self.slider_row, column=0, pady=5, padx=10, sticky="w")
        slider = Scale(self.master, from_=from_, to=to_, orient=tk.HORIZONTAL,fg="white",bg="brown",activebackground="Coral",borderwidth=2,cursor="sb_h_double_arrow")
        slider.set(default)
        slider.grid(row=self.slider_row, column=1, pady=5, padx=10, sticky="ew")
        slider.bind("<ButtonRelease-1>", lambda event, cmd=command: cmd(event))
        self.slider_row += 1
    


    def apply_thresholding_segmentation(self):
        threshold_value = 127
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        self.update_image(cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR))
        self.slider_row = 2
        self.add_slider(1, 250, 127, self.update_thresholding_segmentation)


    def update_thresholding_segmentation(self, event):
        threshold_value = int(event.widget.get())
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        self.update_image(cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR))




    def apply_region_split_merge_segmentation(self):
        region_split_merge_image = self.original_image.copy()
        height, width = region_split_merge_image.shape[:2]

        def region_growing(image, seed):
            visited = set()
            stack = [seed]

            while stack:
                x, y = stack.pop()
                if (x, y) not in visited:
                    visited.add((x, y))
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if abs(int(image[y, x]) - int(image[ny, nx])) < 20:
                                    stack.append((nx, ny))

            return visited

        def merge_regions(visited_regions):
            new_regions = []
            for region in visited_regions:
                merged_region = region.copy()
                while True:
                    for other_region in visited_regions:
                        if other_region != region and not set(region).isdisjoint(other_region):
                            merged_region.update(other_region)
                            visited_regions.remove(other_region)
                            break
                    else:
                        break
                new_regions.append(merged_region)
            return new_regions

        seeds = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

        visited_regions = []
        for seed in seeds:
            visited_region = region_growing(region_split_merge_image, seed)
            visited_regions.append(visited_region)

        merged_regions = merge_regions(visited_regions)

        for region in merged_regions:
            for x, y in region:
                region_split_merge_image[y, x] = 255

        self.update_image(region_split_merge_image)




    def apply_hough_circle_transform(self):
        # Convert the original image to grayscale
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # Detect circles using Hough circle transform with specified parameters
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        # Check if any circles are detected
        if circles is not None:
            # Convert the circle parameters to integer
            circles = np.uint16(np.around(circles))
            # Create a copy of the original image for drawing circles
            hough_image = self.original_image.copy()
            # Draw detected circles on the image
            for i in circles[0, :]:
                cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the outer circle
                cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)       # Draw the center of the circle
            # Update the image display with the circles drawn
            self.update_image(hough_image)



        

    def open_page3(self):
            os.system("python Page3.py") 


root = Tk()
app = ImageProcessingApp(root)
root.mainloop()
