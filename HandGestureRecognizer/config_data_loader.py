import os
import sys
import json
import torch 

def get_model_gestures():
    # Define gesture names
    from HandGestureRecognizer.model_architecture import ConvColumn
    with open('config.json') as f:
        config = json.load(f)

    with open(config['class_mappings']) as f:
        gesture_classes = {ind : class_name.strip() for ind, class_name in enumerate(f.readlines())}

    model = ConvColumn(27,(3, 3, 3))
    model.load_state_dict(torch.load(config['model_weights'], map_location='cpu')['state_dict'])
    return gesture_classes, model

if __name__ == "__main__":
    get_model_gestures()