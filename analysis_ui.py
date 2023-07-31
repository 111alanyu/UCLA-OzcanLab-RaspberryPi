import tkinter as tk
from tkinter import ttk
from quantify_VFA_timelapse_decap import *
from PIL import ImageTk, Image
from csv_analysis_code import * 
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
        
        self.next_analysis = tk.Button(self, text = "Next Step", command = self.next_step)
        self.next_analysis.grid(row = 2, column = 4, columnspan = 2, sticky = "nsew")
        
        self.image = Image.open("/home/pi/Desktop/campus-seal.jpg")
        self.resized_image = self.image.resize((500,400))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.grid(row = 3, column = 1, columnspan = 3)
        
        self.next_img = tk.Button(self, text = "Next", command = self.change_image)
        self.next_img.grid(row = 3, column = 4, columnspan = 1)
        
        self.next_img = tk.Button(self, text = "Prev", command = self.change_image_back)
        self.next_img.grid(row = 3, column = 0, columnspan = 1)
        
        self.curr_count = 0
        
        self.output_path = None
        
    def next_step(self):
        print("Reflog")
        
        sample_folder = self.output_path
        a_output_path = self.directory + "/quality_control"
        
        if not os.path.exists(a_output_path):
            os.mkdir(a_output_path)
        
        features_folder = 'predicted_concentration_features'
        prediction_folder = 'pred'
        encapsulate(sample_folder, a_output_path, prediction_folder, features_folder, "mean_r=30_fixed", "std_r=30_fixed")
    
    def convert_to_tiff(self):
        self.directory = self.capture_object.get_file()
        
        print("YUHJ", self.directory)
        brightnesses = [1]

        print(self.directory)
        if(not self.directory):
            print("Capture was not ran")
                    
        for file in glob.glob(self.directory + '/*.jpg'):
            print(file)
            os.system("dcraw -v -w -o 1 -W -b " + str(brightnesses[0]) + " -q 3 -6 -T " + file)
            
        
    
    def ui_holder(self):
        self.file_path = self.path_entry.get()
        if (not self.file_path):
            if(not self.directory):
                print("No file")
            self.file_path = self.directory
        
        print("$$$$" + (str)(self.file_path))
        
        print("~~~~" + (str)(self.capture_object.getinTimeBc))
        
        self.output_path = main_ui_analysis(self.file_path, self.capture_object.getinTimeBc())
        
        print("%%%%" + self.output_path)
        
        self.process_image = self.file_path + "/processed_jpgs"
        self.image_address = [os.path.join(self.process_image, file) for file in os.listdir(self.process_image)]
        
        self.image = Image.open(self.image_address[0])
        self.resized_image = self.image.resize((500,500))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.grid(row = 3, column = 1, columnspan = 3)
        
    def change_image(self):
        if(self.image_address == None):
            print("No Images to Display")
        
        if self.curr_count < len(self.image_address) - 1:
            self.curr_count += 1
        
        self.image = Image.open(self.image_address[self.curr_count % len(self.image_address)])
        self.resized_image = self.image.resize((500,500))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.grid(row = 3, column = 1, columnspan = 3)
        
    def change_image_back(self):
        if(self.image_address == None):
            print("No Images to Display")
        if self.curr_count > 0: 
            self.curr_count -= 1
        
        self.image = Image.open(self.image_address[self.curr_count])
        self.resized_image = self.image.resize((500,500))
        self.image_tk = ImageTk.PhotoImage(self.resized_image)
        self.image_label = tk.Label(self, image=self.image_tk)
        self.image_label.grid(row = 3, column = 1, columnspan = 3)
       
    def pool_input(self):
        self.path_val = self.path_entry.get()
        self.segment_val = self.segment_entry.get()
        
        print(self.path_val)
        print(self.segment_val)
    
    def checkbox_clicked(self):
        print("Hello World")
        
        
        
        
        