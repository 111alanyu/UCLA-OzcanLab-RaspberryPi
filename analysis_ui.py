import tkinter as tk
from tkinter import ttk


class MyFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        label = tk.Label(self, text = "This is a test")
        label.pack()
        