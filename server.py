import socket
import cv2
import pickle
from HandGestureRecognizer.model_architecture import ConvColumn
import torch
import numpy as np
import os
import sys
import json

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
    HOST = socket.gethostbyname(socket.gethostname())
    PORT = 11999
    server_sock.bind((HOST, PORT))
    server_sock.listen()
    CLOSE_SERVER = False

    script_relative_path = sys.argv[0].strip('.\\').split('\\')[:-1] + ['HandGestureRecognizer']
    script_relative_path = '/'.join(script_relative_path)

    # Define gesture names
    with open(os.path.join(script_relative_path, 'config.json')) as f:
        config = json.load(f)

    with open(os.path.join(script_relative_path, config['class_mappings'])) as f:
        gesture_classes = {ind : class_name.strip() for ind, class_name in enumerate(f.readlines())}

    model = ConvColumn(27,(3, 3, 3))
    model_weights = torch.load(os.path.join(script_relative_path, config['model_weights']), map_location='cpu')['state_dict']
    model_weights = {layer_name.strip('module.') : weight for layer_name, weight in model_weights.items()}
    model.load_state_dict(model_weights)
    print('model loaded')
    # output = model(Variable(data).unsqueeze(0))
    # out = (output.data).cpu().numpy()[0]
    # print('Model output:', out)
    # indices = np.argmax(out)
    while True:
        comm_socket, addr = server_sock.accept()
        print('Connection Accepted')
        while True:
            try:
                frame = b""
                data_size = comm_socket.recv(1024)
                if data_size == b'CLOSE SERVER':
                    CLOSE_SERVER = True
                    break
                data = comm_socket.recv(int(data_size.decode()), socket.MSG_WAITALL)
                image_frames = pickle.loads(data)
                print(image_frames.shape)
                prediction = model(image_frames.unsqueeze(0)).data.numpy()[0]
                prediction = np.argmax(prediction)
                comm_socket.send(gesture_classes[prediction].encode())
            except Exception as e:
                print(e)
                CLOSE_SERVER = True
                break
        if CLOSE_SERVER:
            break