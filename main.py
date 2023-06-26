import tkinter as tk
from tkinter import ttk
from cameraCaptrure_timelapse_background import *
from quantify_VFA_timelapse_decap import *
from analysis_ui import *
import argparse

def main(): 
    # root window
    root = tk.Tk()
    root.geometry('500x1000')
    root.title('Notebook Demo')

    # create a notebook
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, anchor = "nw")

    ap = argparse.ArgumentParser()

    ap.add_argument("-o", "--output", required=True,

    help="path to output directory to store snapshots")

    ap.add_argument("-p", "--picamera", type=int, default=-1,

    help="whether or not the Raspberry Pi camerashould be used")

    args = vars(ap.parse_args())
    
    pba = PhotoBoothApp(args["output"], root, args)
    frame1 = pba.getFrame()
    
    frame2 = MyFrame(notebook)
    
    frame1.pack(fill='both', expand=True)

    # add frames to notebook
    

    notebook.add(frame1, text='Capture')
    notebook.add(frame2, text='Analysis')

    notebook.pack(fill="both", expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()