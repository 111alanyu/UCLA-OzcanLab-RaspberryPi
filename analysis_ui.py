import tkinter as tk
from tkinter import ttk


class MyFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.exp_val = None
        self.gap_val = None
        self.num_of_imgs_val = None
        self.path_val = None
        
        self.exp = tk.Label(self, text = "Exp")
        self.exp.pack()
        self.exp_entry = tk.Entry(self)
        self.exp_entry.pack()
        
        self.gap = tk.Label(self, text = "Gap")
        self.gap.pack()
        self.gap_entry = tk.Entry(self)
        self.gap_entry.pack()
        
        self.num_of_imgs = tk.Label(self, text = "Number of Images")
        self.num_of_imgs.pack()
        self.num_of_imgs_entry = tk.Entry(self)
        self.num_of_imgs_entry.pack()
        
        self.path = tk.Label(self, text = "Path")
        self.path.pack()
        self.path_entry = tk.Entry(self)
        self.path_entry.pack()
        
        self.begin_anal_button = tk.Button(self, text = "Begin Analysis", command = self.pool_input)
        self.begin_anal_button.pack()
        
    def pool_input(self):
        self.exp_val = self.exp_entry.get()
        self.gap_val = self.gap_entry.get()
        self.num_of_imgs_val = self.num_of_imgs_entry.get()
        self.path_val = self.path_entry.get()
        
        print(self.exp_val)
        print(self.gap_val)
        print(self.num_of_imgs_val)
        print(self.path_val)
        
        
        
        
        