import numpy as np
import pickle
from PIL import Image
import torch
from torchvision import transforms as transforms
from torch.utils.data import Dataset
import random


class BCDataset(Dataset):
    def __init__(self, pickle_file='/root/xarm/fixed_final_data.pkl'):
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        self.pickle_file = pickle_file
        self.action_scale = 30
    def __len__(self):
        data = open(self.pickle_file, 'rb')
        data = pickle.load(data)
        return len(data)

    def __getitem__(self, idx):
        data = open(self.pickle_file, 'rb')
        data = pickle.load(data)[idx]
        step_idx = random.randrange(5)
        image = data['images'][step_idx]
        action = data['actions'][step_idx]
        im = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        im = Image.fromarray(im)
        image_tensor = self.transform(im)
        
        action_tensor = torch.FloatTensor(np.array(action, dtype=np.float32)) / self.action_scale
        
        return image_tensor, action_tensor