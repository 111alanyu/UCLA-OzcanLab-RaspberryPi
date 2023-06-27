import tkinter as tk
from tkinter import ttk
from quantify_VFA_timelapse_decap import *
from PIL import ImageTk, Image
import os


class MyFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.path_val = None
        self.segment_val = None
        self.checkbox = None
        
        self.path = tk.Label(self, text = "Path")
        self.path.pack()
        self.path_entry = tk.Entry(self)
        self.path_entry.pack()
        
        self.segment = tk.Label(self, text = "Segment")
        self.segment.pack()
        self.segment_entry = tk.Entry(self)
        self.segment_entry.pack()
        
        self.checkbox_var = tk.IntVar()
        self.label = tk.Label(self, text = "Use from last?")
        self.checkbox = tk.Checkbutton(self, variable = self.checkbox_var, command = self.checkbox_clicked)
        self.label.pack()
        self.checkbox.pack()
        
        self.begin_anal_button = tk.Button(self, text = "Begin Analysis", command = self.ui_holder)
        self.begin_anal_button.pack()
        
        
        self.image = Image.open("/home/pi/Desktop/campus-seal.jpg")
        self.resized_image = self.image.resize((250,200))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.pack()
        
        
    def ui_holder(self):
        main_ui_analysis(self.path_entry.get())
        self.process_image = self.path_entry.get() + "/processed_jpgs"
        image_address = [os.path.join(self.process_image, file) for file in os.listdir(self.process_image)]
        
        self.image = Image.open(image_address[0])
        self.resized_image = self.image.resize((250,200))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.pack()
        
    def pool_input(self):
        self.path_val = self.path_entry.get()
        self.segment_val = self.segment_entry.get()
        
        print(self.path_val)
        print(self.segment_val)
    
    def checkbox_clicked(self):
        print("Hello World")
        
        
        
        
        