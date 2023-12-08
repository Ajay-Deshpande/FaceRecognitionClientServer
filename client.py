import socket
import cv2
from collections import deque
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from torchvision.transforms.v2 import Compose, ToTensor, CenterCrop
from PIL import Image
import pickle
import torch
import numpy as np
# import os
# import sys
# import json

class CustomSharedDataManager(BaseManager):
    pass

class SharedDataClassStructure():
    def __init__(self):
        self.frame_buffer = deque([], maxlen = 18)
        self.prediction = "Doing Other Things"
        self.video_camera_active = True
        self.frame_buffer_size = len(self.frame_buffer)

    def set_prediction(self, prediction):
        self.prediction = prediction
        return
    
    def get_frame_buffer_size(self):
        return self.frame_buffer_size
    
    def append_to_frame_buffer(self, latest_frame):
        self.frame_buffer.append(latest_frame)
        self.frame_buffer_size = len(self.frame_buffer)
        return
    
    def stop_video_camera(self):
        self.video_camera_active = False
        return
    
    def get_video_camera_state(self):
        return self.video_camera_active
    
    def get_frame_buffer(self):
        return tuple(self.frame_buffer)

    def get_prediction(self):
        return self.prediction

def create_video_buffer(shared_data):
    video_camera = cv2.VideoCapture(0)
    transform = Compose([CenterCrop(84), ToTensor()])
    while shared_data.get_video_camera_state():
        
        _, latest_frame = video_camera.read()
        
        image_frame = cv2.resize(latest_frame, (149, 84))
        image_frame = Image.fromarray(image_frame.astype('uint8'), 'RGB')
        image_frame = transform(image_frame).unsqueeze(0)
        shared_data.append_to_frame_buffer(image_frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(latest_frame, shared_data.get_prediction(), (40,40), font, 1, (0,0,0), 2)
        cv2.imshow('preview', latest_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared_data.stop_video_camera()
            break
    cv2.destroyAllWindows()
    return
        
    

def main():
    HOST = socket.gethostbyname(socket.gethostname())
    PORT = 11999
    
    CustomSharedDataManager.register('SharedDataClass', SharedDataClassStructure)
    with CustomSharedDataManager() as manager:
        shared_data = manager.SharedDataClass()
        task = Process(target = create_video_buffer, args = (shared_data, ))
        task.start()

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((HOST, PORT))
        print('Connected')
            
        while shared_data.get_frame_buffer_size() < 18:
            continue
        print('Starting client send')
        while shared_data.get_video_camera_state():
            image_transform = torch.cat(shared_data.get_frame_buffer()).permute(1, 0, 2, 3)
            binary_image_transform = pickle.dumps(image_transform)
            # Send size of data
            client_sock.sendall(f"{len(binary_image_transform)}".encode())
            client_sock.sendall(binary_image_transform)
            prediction = client_sock.recv(1024)
            shared_data.set_prediction(prediction.decode())
        client_sock.send('CLOSE SERVER'.encode())
        task.join()
        client_sock.close()
    return 
    
if __name__ == "__main__":
    main()