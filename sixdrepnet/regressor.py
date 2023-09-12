from math import cos, sin

import torch
from torch.hub import load_state_dict_from_url
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

from model import SixDRepNet
import utils


class SixDRepNet_Detector():

    def __init__(self, gpu_id : int=0, dict_path: str=''):
        """
        Constructs the SixDRepNet instance with all necessary attributes.

        Parameters
        ----------
            gpu:id : int
                gpu identifier, for selecting cpu set -1
            dict_path : str
                Path for local weight file. Leaving it empty will automatically download a finetuned weight file.
        """

        self.gpu = gpu_id
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
                                backbone_file='',
                                deploy=True,
                                pretrained=False)
        # Load snapshot
        if dict_path=='':
            saved_state_dict = load_state_dict_from_url("https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth")    
        else:
            saved_state_dict = torch.load(dict_path)

        self.model.eval()
        self.model.load_state_dict(saved_state_dict)
        
        if self.gpu != -1:
            self.model.cuda(self.gpu)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def predict(self, img):
        """
        Predicts the persons head pose and returning it in euler angles.

        Parameters
        ----------
        img : array 
            Face crop to be predicted

        Returns
        -------
        pitch, yaw, roll
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transformations(img)

        img = torch.Tensor(img[None, :])

        if self.gpu != -1:
            img = img.cuda(self.gpu)
     
        pred = self.model(img)
                
        euler = utils.compute_euler_angles_from_rotation_matrices(pred)*180/np.pi
        p = euler[:, 0].cpu().detach().numpy()
        y = euler[:, 1].cpu().detach().numpy()
        r = euler[:, 2].cpu().detach().numpy()

        return p,y,r


    def draw_axis(self, img, yaw, pitch, roll, x,y,w,h, value, tdx=None, tdy=None, size = 100):
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        img : array
            Target image to be drawn on
        yaw : int
            yaw rotation
        pitch: int
            pitch rotation
        roll: int
            roll rotation
        tdx : int , optional
            shift on x axis
        tdy : int , optional
            shift on y axis
            
        Returns
        -------
        img : array
        """

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = x
            tdy = y
        else:
            height, width = img.shape[:2]
            tdx = w / 2
            tdy = h / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (w/200) * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (w/200) * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (w/200) * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (w/200) * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (w/200) * (sin(yaw)) + tdx
        y3 = size * (w/200) * (-cos(yaw) * sin(pitch)) + tdy
    
        if value:
            cv2.line(img, (int(x), int(y)), (int(x + x1),int(y + roll[0]*size * (w/200))),(0,0,255),4) 
            cv2.line(img, (int(x + x2 - tdx), int(y + tdy*2)), (int(x + x1 + x2 - tdx),int(y + roll[0]*size * (w/200) + tdy*2)),(0,0,255),4) 
            cv2.line(img, (int(x), int(y)), (int(x + x2 - tdx),int(y + tdy*2)),(0,0,255),4)
            cv2.line(img, (int(x + x1),int(y + roll[0]*size * (w/200))), (int(x + x1 + x2 - tdx), int(y + tdy*2 + roll[0]*size * (w/200))),(0,0,255),4) 

            cv2.line(img, (int(x), int(y)), (int(x + yaw[0]*size * (w/200)),int(y - pitch[0] * size * (w/200))),(255,0,0),4) #yaw
            cv2.line(img, (int(x + x2 - tdx), int(y + tdy + tdy)), (int(x + x2 - tdx + yaw[0]*size * (w/200)),int(y + tdy + tdy - pitch[0] * size * (w/200))),(255,0,0),4) #yaw
            cv2.line(img, (int(x + x1), int(y + roll[0]*size * (w/200))), (int(x + x1 + yaw[0]*size * (w/200)),int(y - pitch[0] * size * (w/200) + roll[0]*size * (w/200))),(255,0,0),4) #yaw
            cv2.line(img, (int(x + x1 + x2 - tdx), int(y + tdy*2 + roll[0]*size * (w/200))), (int(x + x1 + x2 - tdx + yaw[0]*100),int(y + tdy*2 - pitch[0] * size * (w/200) + roll[0]*100)),(255,0,0),4) #yaw

            cv2.line(img, (int(x + yaw[0]*size * (w/200)), int(y - pitch[0] * size * (w/200))), (int(x + x1 + yaw[0]*size * (w/200)),int(y + roll[0]*size * (w/200) - pitch[0] * size * (w/200))),(0,255,0),4) 
            cv2.line(img, (int(x + x2 - tdx + yaw[0]*size * (w/200)), int(y + tdy*2 - pitch[0] * size * (w/200))), (int(x + x1 + x2 - tdx + yaw[0]*size * (w/200)),int(y + roll[0]*size * (w/200) + tdy*2 - pitch[0] * size * (w/200))),(0,255,0),4) 
            cv2.line(img, (int(x + yaw[0]*size * (w/200)), int(y - pitch[0] * size * (w/200))), (int(x + x2 - tdx + yaw[0]*size * (w/200)),int(y + tdy*2 - pitch[0] * size * (w/200))),(0,255,0),4)
            cv2.line(img, (int(x + x1 + yaw[0]*size * (w/200)),int(y + roll[0]*size * (w/200) - pitch[0] * size * (w/200))), (int(x + x1 + x2 - tdx + yaw[0]*size * (w/200)), int(y + tdy*2 + roll[0]*size * (w/200) - pitch[0] * size * (w/200))),(0,255,0),4) 

        height, width, channels = img.shape[:3]
        threshold = 0.1
        text = ['', '', '']
        color = [(255, 0, 0),(255, 0, 0),(255, 0, 0)]
        data = [round(pitch[0], 3), round(yaw[0], 3), round(roll[0], 3)]
        output = [-1,-1,-1]  #0:ok, 1:minus, 2:plus
        flg=True
        cnt = 0
        for i in range(3):
            if abs(data[i]) <= threshold:
                text[i] = 'OK!'
                color[i] = (255, 0, 0)
                output[i] = 0
            else:
                color[i] = (0, 0, 255)
                text[i] = 'BAD...'
                flg=False
                if data[i] < 0:
                    output[i] = 1
                else :
                    output[i] = 2
        
        if value:
            cv2.rectangle(img, (int(width*0/20), int(height*6.3/10)), (int(width*5/20), int(height*8.5/10)), (255,255,255), thickness=-1)
            cv2.rectangle(img, (int(width*0), int(height*6.3/10)), (int(width*5/20), int(height*8.5/10)), (0,0,0))

            #pitch
            cv2.putText(img,text='1: '+ text[0],org=(10, int(height*6.8/10)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,color=color[0],thickness=2,lineType=cv2.LINE_AA)
            #yaw
            cv2.putText(img,text='2: '+ text[1],org=(10, int(height*7.5/10)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,color=color[1],thickness=2,lineType=cv2.LINE_AA)   
            #roll 
            cv2.putText(img,text='3: '+ text[2],org=(10, int(height*8.2/10)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,color=color[2],thickness=2,lineType=cv2.LINE_AA)
    

        return img, flg, output


