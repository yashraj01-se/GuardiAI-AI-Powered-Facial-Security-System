#Import Kivy Dependencies:
from kivy.app import App #Base app class
from kivy.uix.boxlayout import BoxLayout #Basic layout of our kivy app
from kivy.uix.image import Image as KivyImage #UX component
from kivy.uix.button import Button #UX component
from kivy.uix.label import Label #UX componenet
from kivy.clock import Clock #Needed for Real Time update
from kivy.graphics.texture import Texture #OpenCV image->Texture , Image->Texture
from kivy.logger import Logger #see some metrics
from kivy.graphics import Color, Rectangle #for color change

#Import other Dependencies:
import os
import cv2
import time
import torch
import subprocess
import numpy as np
import pygetwindow as gw
from PIL import Image 
from layers import L1Dist
from speech_helper import speak
from torchvision import transforms
from Saimese_model import ModelSaimese,SiameseNetwork

count=5

#Building the App and its Layout
class CamApp(App):

    def build(self): #important to inherit just as forward pass in Pytorch
        #making the Main Components of our App:
        self.header = Label(text="GuardiAI-Face Recognition Verification System",size_hint=(1, 0.1),font_size='35sp',bold=True,color=(1, 1, 1, 1))  # black text
        self.web_cam=KivyImage(size_hint=(1,.8)) #image Portion
        self.button=Button(text="Verify",on_press=self.verify, size_hint=(1,.1)) #Button to initiate Action
        self.Verification_label=Label(text="Verification Uninitiated", size_hint=(1,.1)) #TextBox (Three Status: Verification Unitiated, Verified , Unverified)
        
        #adding the Components to our Layout:
        layout=BoxLayout(orientation='vertical') #our App layout Basically set to be vertical
        layout.add_widget(self.header) #header
        layout.add_widget(self.web_cam) #Adding image component
        layout.add_widget(self.button) #Adding Button component
        layout.add_widget(self.Verification_label) #Adding verification(Text) component

        # Default color (white)
        self.color = (255, 255, 255)

        #Load The pytorch_model:
        self.Model_full = torch.load("siamese_model_final_path_model.pth", map_location="cpu", weights_only=False)
        self.Model_full.eval()

        #Setup of the Video Capture Device(WebCam):
        self.capture=cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0) #Scheduling our webcam

        return layout #return the whole layout while Running the app
    
    #------------------------------------------------------------------------------------------------------------------------------------#

    #The Continuously Running Update method basically for continuous feed:
    def update(self,*args):
        
        #Frame for our webcam:
        ret,frame=self.capture.read()

        # Draw rectangle
        start_point = (200, 120)
        end_point = (200 + 250, 120 + 250)
        thickness = 3
        cv2.rectangle(frame, start_point, end_point,self.color, thickness)


        #Flip Horizontal and convert Image to texture:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR → RGB
        frame = cv2.flip(frame,0)# Flip frame vertically
        buf = frame.tobytes()# Coversion of image to buffer
        img_texture=Texture.create(size=(frame.shape[1], frame.shape[0]),colorfmt='rgb') #Creating the texture type
        img_texture.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte') #conversion of buffer to image texture
        self.web_cam.texture=img_texture
        #basically Converting our raw OpenCV image array to a texture for rendering.
        #the setting our image equal to that texture

    #------------------------------------------------------------------------------------------------------------------------------------#
    
    # Preprocessing images:
    def preprocess(self,image_path):
        transform = transforms.Compose([
        transforms.Resize((105, 105)), #resizing
        transforms.ToTensor() #coverting to tensor
    ])
        image = Image.open(image_path).convert('RGB')#Opening the image
        return transform(image).unsqueeze(0) #passing it through the transform and returning it
    
    #------------------------------------------------------------------------------------------------------------------------------------#

    def reset_verification_label(self, dt):
        self.Verification_label.text = "Verification Uninitiated"
        self.Verification_label.color = (255, 255, 255) 
        self.color =(255, 255, 255) 

    #Verification Function
    def verify(self,*args):
        global count
       
        #specifying the Condition for our application to Verify a Person:
        verification_threshold=0.50
        detection_threshold=0.85

        #Capture Input image from web cam to Save into Input image folder:
        SAVE_PATH=os.path.join('Application_Data', 'Input_Image', 'input_image.jpg')
        ret,frame=self.capture.read()
        frame=frame[120:120+250,200:200+250,:]
        cv2.imwrite(SAVE_PATH,frame)

        results = []
        for image_name in os.listdir(os.path.join('Application_Data', 'Verification_Images')):
            input_image = self.preprocess(os.path.join('Application_Data', 'Input_Image', 'input_image.jpg')) #webcam Image
            validation_img = self.preprocess(os.path.join('Application_Data', 'Verification_Images', image_name)) #Verfication Images

            with torch.inference_mode():
                result =self.Model_full(input_image, validation_img)
                results.append(result.item())

        #Detection threshold: how many predictions are considered positive
        detection = sum(r > detection_threshold for r in results)

        #Verification threshold: proportion of positives
        verification = detection / len(results)

        #Boolean verification
        verified = verification > verification_threshold



        if verified:
            self.Verification_label.text = "Yashraj Sharma!! Welcome Back" #label Changes
            self.Verification_label.color = (0, 1, 0, 1)   # Green text
            self.color = (0, 255, 0)   # Green when verified
            speak("Yashraj Sharma Welcome Back")
            count=5

            # Open Notepad
            proc = subprocess.Popen(['notepad.exe'])

        else:
            count-=1
            if count==0:
                self.Verification_label.text = "Notepad Locked"
                self.Verification_label.color = (1, 0, 0, 1)   # Red text
                self.color = (0, 0, 255) 
                speak("Notepad Locked")
                self.button.disabled = True
            else:
                self.Verification_label.text = "Unverified: Access Denied"
                self.Verification_label.color = (1, 0, 0, 1)   # Red text
                self.color = (0, 0, 255) 
                speak("Unverified Access Denied")

        Clock.schedule_once(self.reset_verification_label, 10)

        
        print("detection:", detection)
        print("verification:", verification)
        print("verification threshold:", verification_threshold)
        print("verified check:", verification > verification_threshold)
        
        return results, verified


if __name__=="__main__": #smooth running of application
    CamApp().run()