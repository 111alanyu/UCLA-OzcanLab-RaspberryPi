# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:30:03 2022

@author: DL9
"""
#minor change

# import the necessary packages

from __future__ import print_function

from PIL import Image as Imgp

from PIL import ImageTk

import tkinter as tki

from tkinter import *

import threading 

import datetime

import imutils

import cv2

import os





# from pyimagesearch.photoboothapp import PhotoBoothApp

from imutils.video import VideoStream

import argparse

import time

import RPi.GPIO as GPIO #MUST CONNECT GND LED pin to ic pins. 
import time



# construct the argument parse and parse the arguments/home/pi/cameraCaptrure_timelapse_background.py


# initialize the video stream and allow the camera sensor to warmup



# vs = cv2.VideoCapture(-1, cv2.CAP_V4L)
# vs = VideoStream(-1).start()

time.sleep(2.0)


class PhotoBoothApp:
    
    def setup(self) :
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sdi, GPIO.OUT)
        GPIO.setup(self.clk, GPIO.OUT)
        GPIO.setup(self.le, GPIO.OUT)
        GPIO.setup(self.oe, GPIO.OUT)
        

    def toggleLE(self):
        GPIO.output(self.le, 1)
        time.sleep(self.clock/1000000.0)
        GPIO.output(self.le, 0)
        time.sleep(self.clock/1000000.0)

    def toggleclk(self):
        GPIO.output(self.clk, 0)
        time.sleep(self.clock/1000000.0)
        GPIO.output(self.clk, 1)
        time.sleep(self.clock/1000000.0)

    def write(self, n):
        GPIO.output(self.sdi, n == 1)
        self.toggleclk()
        GPIO.output(self.sdi, n == 1)
        self.toggleclk()
        GPIO.output(self.sdi, n == 1)
        self.toggleclk()
        GPIO.output(self.sdi, n == 1)
        self.toggleclk()
        GPIO.output(self.sdi, n == 2)
        self.toggleclk()
        GPIO.output(self.sdi, n == 2)
        self.toggleclk()
        GPIO.output(self.sdi, n == 2)
        self.toggleclk()
        GPIO.output(self.sdi, n == 2)
        self.toggleclk()
        self.toOutput()

    def toOutput(self):
        self.toggleLE()
        GPIO.output(self.oe, 0)
        time.sleep(2)
        #GPIO.output(self.oe, 1)
        
    def get_file(self):
        return self.current_file

    def __init__(self, outputPath, parent, args):

        # store the video stream object and output path, then initialize

        # the most recently read frame, thread for reading frames, and

        # the thread stop event
        
        
        self.current_file = None
        
        self.sdi = 5
        self.clk = 6
        self.le = 26
        self.oe = 16
        self.clock = 200
        self.args = args
        GPIO.setwarnings(False)

        self.vs = VideoStream(usePiCamera=args["picamera"] >0).start()
        self.parent = parent

        self.setup()
        

        self.outputPath = outputPath

        self.frame = None

        self.thread = None

        self.stopEvent = None


        self.picNum = 0
        # initialize the root window and image panel

        self.panel = None



        self.panelR = Frame(self.parent)

        self.panelR.pack(side = RIGHT )

        # create a button, that when pressed, will take the current

        # frame and save it to file

        l = Label(self.panelR, text="Exposure(s):")

        l.pack(side=TOP)

        self.txtExpT = tki.Text(self.panelR, height = 5, width = 25)

        self.txtExpT.pack(side = TOP)

        self.txtExpT.insert(END, "0.0001")

        

        l = Label(self.panelR, text="Number of Images:")

        l.pack(side=TOP)

        self.txtNumP = tki.Text(self.panelR, height = 1, width = 25)

        self.txtNumP.pack(side = TOP)

        self.txtNumP.insert(END, "8")


        l = Label(self.panelR, text="Time Between Capture(s):")

        l.pack(side=TOP)

        self.txtTimeBC = tki.Text(self.panelR, height = 1, width = 25)

        self.txtTimeBC.pack(side = TOP)

        self.txtTimeBC.insert(END, "30")
        

        btn = tki.Button(self.panelR, text="Start Capture",

        command=self.takeSnapshot)

        btn.pack(side="bottom", fill="both", expand="yes", padx=10,

        pady=10)

        btn2 = tki.Button(self.panelR, text="Control Capture",

        command=self.takeControl)

        btn2.pack(side="bottom", fill="both", expand="yes", padx=10,

        pady=10)
        
        


        # start a thread that constantly pools the video sensor for

        # the most recently read frame

        self.stopEvent = threading.Event()

        self.thread = threading.Thread(target=self.videoLoop, args=())

        self.thread.start()



        # set a callback to handle when the window is closed

        self.parent.wm_title("PyImageSearch PhotoBooth")

        self.parent.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        

    def videoLoop(self):  #ALAN THIS IS WHERE THEY DECLARE THE VIDEO!!!
        #TODO: Make it show up in the correct location. 
        self.write(1)

        # DISCLAIMER:

        # I'm not a GUI developer, nor do I even pretend to be. This

        # try/except statement is a pretty ugly hack to get around

        # a RunTime error that Tkinter throws due to threading

        try:

            # keep looping over frames until we are instructed to stop

            while not self.stopEvent.is_set():

                # grab the frame from the video stream and resize it to

                # have a maximum width of 300 pixels

                self.frame = self.vs.read()

                self.frame = imutils.resize(self.frame, width=500)



                # OpenCV represents images in BGR order; however PIL

                # represents images in RGB order, so we need to swap

                # the channels, then convert to PIL and ImageTk format

                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                image = Imgp.fromarray(image)

                image = ImageTk.PhotoImage(image)



                # if the panel is not None, we need to initialize it

                if self.panel is None:

                    self.panel = tki.Label(image=image)

                    self.panel.image = image

                    self.panel.pack(side="left", padx=10, pady=10)



                # otherwise, simply update the panel

                else:

                    self.panel.configure(image=image)

                    self.panel.image = image



        except RuntimeError as e:

            print("[INFO] caught a RuntimeError")


    def takeSnapshot(self):

        # grab the current timestamp and use it to construct the

        # output path

        inNumP = self.txtNumP.get("1.0", "end")

        inExpTxt = self.txtExpT.get("0.0", "end")

        inTimeBc = self.txtTimeBC.get("1.0", "end")

        #expT = int(inExpTxt)*1000*1000

        expT = [float(x.strip()) for x in inExpTxt.split(',')]

        expTBC = [float(x.strip()) for x in inTimeBc.split(',')]

        ts = datetime.datetime.now()

        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        self.stopEvent.set()
        self.write(1)

        self.vs.stream.release()

        expT_rep = 1
        total_time = 0
        
        
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.directory_name = f"output_data_{self.timestamp}"
        self.outputPath = self.outputPath + "/" + self.directory_name
        os.mkdir(self.outputPath)
        self.current_file = self.outputPath 
        
        
        
        print("^^^" + self.outputPath)
        
        
                
        for i in range(int(inNumP)):
            start = time.perf_counter_ns()
            filename = str(i)+ "_" + str(expT[i % expT_rep]) + "_"  +"{}_b1.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            print("@@@" + filename)
            #filename = str(int(expT[i]))+"s_b256.jpg"
            
            p = os.path.sep.join((self.outputPath, filename))         
            print("###" + p)
            if (len(expT) >= int(inNumP)):
                os.system("raspistill -md 0 -bm -ex off -ag 1 -dg 1 --shutter "+str(expT[i % expT_rep]*1000000)+" -st -t 100 -r -o "+p)
                #os.system("dcraw -v -o 1 -W -b 1 -q 3 -6 -T " + p)
                # os.system("convert " + p[:len(p)-4] + ".tiff" + " -separate " + p[:len(p)-4] + "_%d.tiff")
                
                
                
            else:
                
                os.system("raspistill -md 0 -bm -ex off -ag 1 -dg 1 --shutter "+str(expT[i % expT_rep]*1000000)+" -st -t 100 -r -o "+p)
                #os.system("dcraw -v -o 1 -W -b 1 -q 3 -6 -T " + p)
                # os.system("convert " + p[:len(p)-4] + ".tiff" + " -separate " + p[:len(p)-4] + "_%d.tiff")
                
                
#             time.sleep(1)

#             img = ImageTk.PhotoImage(Imgp.open(p))

#             img = img.resize((500,300), Image#.ANTIALIAS)

#             self.panel.configure(image=img)

#             self.panel.image=img

#             self.panel.update()

            print("[INFO] saved {}".format(filename))
            end = time.perf_counter_ns()
            time_evolved = (end-start)/1e9
            total_time = total_time + time_evolved
            print("Evolved_time = ", time_evolved)
            #self.write(0)
            #time.sleep(0.1)
            #time.sleep(int(expTBC[0]))
            sleep_time = int(expTBC[0]) - total_time
            time.sleep(sleep_time)
            total_time = 0
            '''
            if i % expT_rep == 1:
                sleep_time = int(expTBC[0])-total_time
                time.sleep(sleep_time)
                print("Sleep time = ", sleep_time)
                total_time = 0
            else:
                total_time = total_time + 0.1
                time.sleep(0.1)                   
    '''
        
        
        
        self.vs = VideoStream(usePiCamera=self.args["picamera"] > 0).start()
        
        self.stopEvent.clear()

        self.thread = threading.Thread(target=self.videoLoop, args=())

        self.thread.start()

    '''
    These get functions
    '''
    def getFrame(self):
        return self.panelR
    

   
    def takeControl(self):

        # grab the current timestamp and use it to construct the

        # output path

        inNumP = self.txtNumP.get("1.0", "end")

        inExpTxt = self.txtExpT.get("0.0", "end")

        inTimeBc = self.txtTimeBC.get("1.0", "end")

        #expT = int(inExpTxt)*1000*1000
        quantify_VFA_timelapse_decap
        expT = [float(x.strip()) for x in inExpTxt.split(',')]

        expTBC = [float(x.strip()) for x in inTimeBc.split(',')]
        expT = 0.001

        ts = datetime.datetime.now()

        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

        self.stopEvent.set()
        self.write(2)
        time.sleep(0.5)

        self.vs.stream.release()

        time.sleep(0.5)

       # expT_rep = 3
        for i in range(int(1)):

            filename = "0"+str(i)+ "_" + str(expT) + "_"  +"{}_b1.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            
            #filename = str(int(expT[i]))+"s_b256.jpg"
            
            p = os.path.sep.join((self.outputPath, filename))         

            
            os.system("raspistill -md 0 -bm -ex off -ag 16 -dg 1 --shutter "+str(expT*1000000)+" -st -t 100 -r -o "+p)
            #os.system("dcraw -v -o 1 -W -b 1 -q 3 -6 -T " + p)
            # os.system("convert " + p[:len(p)-4] + ".tiff" + " -separate " + p[:len(p)-4] + "_%d.tiff")
                
                
#             time.sleep(1)

#             img = ImageTk.PhotoImage(Imgp.open(p))

#             img = img.resize((500,300), Image#.ANTIALIAS)

#             self.panel.configure(image=img)

#             self.panel.image=img

#             self.panel.update()

            print("[INFO] saved {}".format(filename))
            self.write(1)
            time.sleep(0.1)
            
                             

        self.vs = VideoStream(usePiCamera=args["picamera"] > 0).start()

        time.sleep(0.5)

        self.stopEvent.clear()

        self.thread = threading.Thread(target=self.videoLoop, args=())

        self.thread.start()



    def onClose(self):

        # set the stop event, cleanup the camera, and allow the rest of

        # the quit process to continue

        print("[INFO] closing...")
        
        GPIO.cleanup()
        
        self.stopEvent.set()

        self.vs.stop()

        self.parent.quit()




# start the app

#pba = PhotoBoothApp(vs, args["output"])

#pba.root.mainloop()