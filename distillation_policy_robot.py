import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv

import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import cv2
from distillation_CNNs import CNN_FeatureExtractor_Camera, CNN_FeatureExtractor_Scan, Encoder, Actor_model
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
import datetime


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()

#         self.layer_1 = nn.Linear(state_dim, 800)
#         self.layer_2 = nn.Linear(800, 600)
#         self.layer_3 = nn.Linear(600, action_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, s):
#         s = F.relu(self.layer_1(s))
#         s = F.relu(self.layer_2(s))
#         a = self.tanh(self.layer_3(s))
#         return a


# # TD3 network
# class TD3(object):
#     def __init__(self, state_dim, action_dim):
#         # Initialize the Actor network
#         self.actor = Actor(state_dim, action_dim).to(device)

#     def get_action(self, state):
#         # Function to get the action from the actor
#         state = torch.Tensor(state.reshape(1, -1)).to(device)
#         return self.actor(state).cpu().data.numpy().flatten()

#     def load(self, filename, directory):
#         # Function to load network parameters
#         self.actor.load_state_dict(
#             torch.load("%s/%s_actor.pth" % (directory, filename))
#         )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# # Create the network
# network = TD3(state_dim, action_dim)
# try:
#     network.load(file_name, "./pytorch_models")
# except:
#     raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

transform_camera = Compose([
    Resize((32, 64)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

def normalize_scan_data(data):
    # Find the minimum and maximum values
    min_val = torch.min(data)
    max_val = torch.max(data)
  
    return (data - min_val) / (max_val - min_val)

OUTPUT_DIM = 5
dim = 24 + OUTPUT_DIM + 32

# ASK MURAD ABOUT THE EVAL MODE OR TRAIN MODE
model_camera = CNN_FeatureExtractor_Camera().to(device)
# model_camera.load_model("runs/Nets340713.Mar.16.48/epoch 1185.datcamera_model")
model_camera.load_model("runs/Nets340714.Mar.23.22/epoch 3115.datcamera_model")

model_scan = CNN_FeatureExtractor_Scan().to(device)
# model_scan.load_model("runs/Nets340713.Mar.16.48/epoch 1185.datscan_model")
model_scan.load_model("runs/Nets340714.Mar.23.22/epoch 3115.datscan_model")

model_encoder = Encoder(OUTPUT_DIM).to(device)
# model_encoder.load_model("runs/Nets340713.Mar.16.48/epoch 1185.datencoder_model")
model_encoder.load_model("runs/Nets340714.Mar.23.22/epoch 3115.datencoder_model")

model_actor = Actor_model(dim, 2).to(device)
# model_actor.load_model("runs/Nets340713.Mar.16.48/epoch 1185.datactor_model")
model_actor.load_model("runs/Nets340714.Mar.23.22/epoch 3115.datactor_model")

sucess = 0
failer = 0
iter = 0

# Begin the testing loop
while True:
    # camera_data = env.get_camera_data()
    # camera_data =  torch.stack([
    #         transform_camera(Image.fromarray(camera_data.squeeze().astype(np.uint8)))]).to(device)        
    #  # check if the torch stack is required or not
    # # print transformationed completed
    # print("**********tranformation completed**********")
    
    # features = model_camera(camera_data)
    # print("**********camera features completed**********")
    # encoded_features = model_encoder(features)
    # print("**********encoded features completed**********")
    # #convert the state into tensor and add one more dim for the concatenation
    # state = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(device)
    # concatenated_all_camera = torch.cat((features, encoded_features, state), 1).to(device)
    # print("**********concatenation completed**********")
    # action = model_actor(concatenated_all_camera)
    # print("**********action completed**********")
    # # conver the action output into a numpy array

    # # Convert the action output into a numpy array
    # action = action.cpu().detach().tolist()
    # print("**********action converted to numpy**********")
    # print("printed action: ", action)
    # # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    # a_in = [(action[0][0] + 1) / 2, action[0][1]]

    scan_data = env.get_scan_data()
    #convert the scan data to tensor float 32 and add one more dimension
    scan_data = torch.stack([
            torch.tensor(scan_data.reshape(1, -1), dtype=torch.float32)]).to(device)

    scan_data = normalize_scan_data(scan_data).to(device)
    print("**********scan data completed**********")
    features = model_scan(scan_data)
    print("**********scan features completed**********")
    encoded_features = model_encoder(features)
    print("**********encoded features completed**********")
    #convert the state into tensor and add one more dim for the concatenation
    state = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(device)
    concatenated_all_scan = torch.cat((features, encoded_features, state), 1).to(device)
    print("**********concatenation completed**********")
    action = model_actor(concatenated_all_scan)
    print("**********action completed**********")
    action = action.cpu().detach().tolist()
    print("**********action converted to numpy**********")
    #print("printed action: ", action)
    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0][0] + 1) / 2, action[0][1]]
    print("**********action updated**********")
    print("printed action: ", a_in)
    


    
    # action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity

    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    # On termination of episode
    if done:
        if target:
            print("Target reached")
            sucess += 1
            iter += 1
        else:
            print("Target not reached")
            failer += 1
            iter += 1
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
    
    if iter == 100:
        break

# Print the success and failure rates
print("Success: ", sucess)
print("Failer: ", failer)
print("Success rate: ", sucess/iter)
print("Failer rate: ", failer/iter)


