import tkinter as tk
from tkinter import ttk
from cameraCaptrure_timelapse_background import *
import argparse

def main(): 
    # root window
    root = tk.Tk()
    root.geometry('400x300')
    root.title('Notebook Demo')

    # create a notebook
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, anchor = "nw")

    ap = argparse.ArgumentParser()

    ap.add_argument("-o", "--output", required=True,

    help="path to output directory to store snapshots")

    ap.add_argument("-p", "--picamera", type=int, default=-1,

    help="whether or not the Raspberry Pi camera should be used")


    args = vars(ap.parse_args())
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    
    frame1 = PhotoBoothApp(vs, args["output"], root)
    frame2 = ttk.Frame(notebook, width=400, height=280)


    # create frames


    frame1.pack(fill='both', expand=True)
    frame2.pack(fill='both', expand=True)


    text_label = tk.Label(frame1, text="Capture")
    text_label1 = tk.Label(frame2, text="Analysis")

    text_label.pack()
    text_label1.pack()
    # add frames to notebook

    notebook.add(frame1, text='Capture')
    notebook.add(frame2, text='Analysis')


    root.mainloop()

if __name__ == "__main__":
    main()