import tkinter as tk
from tkinter import ttk
from quantify_VFA_timelapse_decap import *


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
        
    def ui_holder(self):
        main_ui_analysis(self.path_entry.get())
        
    def pool_input(self):
        self.path_val = self.path_entry.get()
        self.segment_val = self.segment_entry.get()
        
        print(self.path_val)
        print(self.segment_val)
    
    def checkbox_clicked(self):
        print("Hello World")
        
        
        
        
        