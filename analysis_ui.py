import tkinter as tk
from tkinter import ttk
from quantify_VFA_timelapse_decap import *
from PIL import ImageTk, Image
import os
import glob



class MyFrame(tk.Frame):
    def __init__(self, parent, capture):
        super().__init__(parent)
        self.path_val = None
        self.segment_val = None
        self.checkbox = None
        
        self.capture_object = capture 
        
        self.columnconfigure(1, minsize = 50)
        
        self.path = tk.Label(self, text = "Path")
        self.path.grid(row = 0, column = 0)
        self.path_entry = tk.Entry(self)
        self.path_entry.grid(row = 0, column = 1, columnspan = 3, sticky = "we")
        
        self.segment = tk.Label(self, text = "Segment")
        self.segment.grid(row = 1, column = 0)
        self.segment_entry = tk.Entry(self)
        self.segment_entry.grid(row = 1, column = 1, columnspan = 3, sticky = "we")
        
        self.columnconfigure(1, weight = 1)
        
        self.tiff_directory = None
        
        '''capture_object
        self.checkbox_var = tk.IntVar()
        self.label = tk.Label(self, text = "Use from last?")
        self.checkbox = tk.Checkbutton(self, variable = self.checkbox_var, command = self.checkbox_clicked)
        self.label.pack()
        self.checkbox.pack()
        '''
        
        self.begin_anal_button = tk.Button(self, text = "Begin Analysis", command = self.ui_holder)
        self.begin_anal_button.grid(row = 2, column = 0, columnspan = 2, sticky = "nsew")
        
        self.convert_jpg_to_tiff = tk.Button(self, text = "Convert Capture .jpg to .tiff", command = self.convert_to_tiff)
        self.convert_jpg_to_tiff.grid(row = 2, column = 2, columnspan = 2, sticky = "nsew")
        
        self.image = Image.open("/home/pi/Desktop/campus-seal.jpg")
        self.resized_image = self.image.resize((500,400))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.grid(row = 3, column = 0, columnspan = 3)
        
        
        
    
    
    def convert_to_tiff(self):
        self.directory = self.capture_object.get_file()
        
        if(not self.directory):
            print("Capture was not ran")
            return
        
        self.img_directory = self.directory + "/img/"
        self.tiff_directory = self.directory + "/tiff/"
        os.mkdir(self.tiff_directory)
        
        
        jpg_files = [f for f in os.listdir(self.img_directory) if f.endswith(".jpg")]
        print(jpg_files)
        
        for i in jpg_files:
            image = Image.open(self.img_directory + i)
            image.save(self.tiff_directory + i[:-4] + ".tiff", "TIFF")
            
        
    
    def ui_holder(self):
        self.file_path = self.path_entry.get()
        if (not self.file_path):
            if(not self.tiff_directory):
                print("No file")
            self.file_path = self.tiff_directory[:-1]
        
        print("$$$$" + (str)(self.file_path))
        
        main_ui_analysis(self.file_path)
        
        self.process_image = self.file_path + "/processed_jpgs"
        image_address = [os.path.join(self.process_image, file) for file in os.listdir(self.process_image)]
        
        self.image = Image.open(image_address[0])
        self.resized_image = self.image.resize((500,500))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.grid(row = 3, column = 0, columnspan = 3)
        
    def pool_input(self):
        self.path_val = self.path_entry.get()
        self.segment_val = self.segment_entry.get()
        
        print(self.path_val)
        print(self.segment_val)
    
    def checkbox_clicked(self):
        print("Hello World")
        
        
        
        
        