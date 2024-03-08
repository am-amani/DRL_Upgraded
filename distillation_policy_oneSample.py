import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import cv2
from distillation_CNNs import CNN_FeatureExtractor_Camera, CNN_FeatureExtractor_Scan, Encoder, Actor
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, copy
from tensorboardX import SummaryWriter
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Initialize empty buffers for T1 and T2


BATCH_SIZE = 1
OUTPUT_DIM = 10
LR = 1e-3
TRAIN_STEP = 0
EPOCH_STEP = 0
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

date = datetime.datetime.now().strftime("%d.%h.%H.%M")

dirPath = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dirPath, "runs/Nets"+str(seed)+str(date))

replay_buffer1 = torch.load('/home/amirmahdi/DRL-robot-navigation/TD3/pytorch_models/replay_buffer_part2.pth')
replay_buffer2 = torch.load('/home/amirmahdi/DRL-robot-navigation/TD3/pytorch_models/replay_buffer_part2.pth')

# Concatenate replay_buffer2 to replay_buffer1
replay_buffer = replay_buffer1 + replay_buffer2

print("Size of final replay buffer:", replay_buffer.size())

def split_replay_buffer(replay_buffer, train_ratio=0.8):
    # Extract all data from the buffer
    all_data = list(replay_buffer.buffer)  # Convert deque to a list for easier handling
    
    # Shuffle the data
    np.random.shuffle(all_data)  # Ensure random order
    
    # Compute split indices based on the train ratio
    total_count = len(all_data)
    train_count = int(total_count * train_ratio)
    test_count = total_count - train_count  # This ensures the correct split
    
    # Split data
    train_data = all_data[:train_count]
    test_data = all_data[train_count:]  # Note: This should automatically give the rest of the data to the test set
    
    # Create new ReplayBuffers for train and test data
    train_buffer = ReplayBuffer(replay_buffer.size())
    test_buffer = ReplayBuffer(replay_buffer.size())
    
    # Populate the new buffers
    for experience in train_data:
        train_buffer.add(*experience)
    for experience in test_data:
        test_buffer.add(*experience)
    
    return train_buffer, test_buffer, train_count, test_count  # Return the actual counts for verification

# Use the corrected function
train_buffer, test_buffer, train_count, test_count = split_replay_buffer(replay_buffer)

transform_camera = Compose([
    Resize((32, 64)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# write a normazliation function for the sample batch of scan data by finding the max and min values of the scan data
def normalize_scan_data(data):
    # Find the minimum and maximum values
    min_val = torch.min(data)
    max_val = torch.max(data)
    print("min_val:", min_val)
    print("max_val:", max_val)

    
    # Normalize the data
    return (data - min_val) / (max_val - min_val)


# ASK MURAD ABOUT THE EVAL MODE OR TRAIN MODE
model_camera = CNN_FeatureExtractor_Camera().to(device)
model_camera_optimizer = optim.Adam(model_camera.parameters(), lr=LR, weight_decay=1e-5)
model_camera_scheduler = optim.lr_scheduler.StepLR(model_camera_optimizer, step_size=1000, gamma=0.1)

model_scan = CNN_FeatureExtractor_Scan().to(device)
model_scan_optimizer = optim.Adam(model_scan.parameters(), lr=LR, weight_decay=1e-5)
model_scan_scheduler = optim.lr_scheduler.StepLR(model_scan_optimizer, step_size=1000, gamma=0.1)

model_encoder = Encoder(OUTPUT_DIM).to(device)
model_encoder_optimizer = optim.Adam(model_encoder.parameters(), lr=LR, weight_decay=1e-5)
model_encoder_scheduler = optim.lr_scheduler.StepLR(model_encoder_optimizer, step_size=1000, gamma=0.1)

model_actor = Actor(546, 2).to(device)
model_actor_optimizer = optim.Adam(model_actor.parameters(), lr=LR, weight_decay=1e-5)
model_actor_scheduler = optim.lr_scheduler.StepLR(model_actor_optimizer, step_size=1000, gamma=0.1)

writer = SummaryWriter('runs/Nets'+str(1)+str(datetime.datetime.now())+'_CNN_'+str(2000))

sample_batch_train_camera = train_buffer.sample_batch(BATCH_SIZE)
sample_batch_train_scan = train_buffer.sample_batch(BATCH_SIZE)
# check the sample batches are the same or not, if same, take a sample from the train buffer again
while np.any(sample_batch_train_camera[0] == sample_batch_train_scan[0]):
    sample_batch_train_camera = train_buffer.sample_batch(BATCH_SIZE)
    sample_batch_train_scan = train_buffer.sample_batch(BATCH_SIZE)

sample_batch_test_camera = test_buffer.sample_batch(BATCH_SIZE)
sample_batch_test_scan = test_buffer.sample_batch(BATCH_SIZE)
# check the sample batches are the same or not, if same, take a sample from the test buffer again
while np.any(sample_batch_test_camera[0] == sample_batch_test_scan[0]):
    sample_batch_test_camera = test_buffer.sample_batch(BATCH_SIZE)
    sample_batch_test_scan = test_buffer.sample_batch(BATCH_SIZE)

# Define the train fucntion
def train(data_batch_camera, data_batch_scan):
    # create a temporary variable to increase +1 after each iteration.
    # This varible will be used to set set either camera data or scan data to the model. 
    # if it is even, then set the camera data to the model, otherwise set the scan data to the model.
    
    #convert the state data to tensor and set it to the device
    state_data_batch_camera = torch.tensor(data_batch_camera[0], dtype=torch.float32).to(device)
    print("action_data_batch_camera.shape:", state_data_batch_camera.shape)
    action_data_batch_camera = torch.tensor(data_batch_camera[1], dtype=torch.float32).to(device)
    print("action_data_batch_camera.shape:", action_data_batch_camera.shape)

    #convert the state data to tensor and set it to the device
    state_data_batch_scan = torch.tensor(data_batch_scan[0], dtype=torch.float32).to(device)
    print("state_data_batch_scan.shape:", state_data_batch_scan.shape)
    action_data_batch_scan = torch.tensor(data_batch_scan[1], dtype=torch.float32).to(device)
    print("state_data_batch_scan.shape:", state_data_batch_scan.shape)


    global TRAIN_STEP
    

    if TRAIN_STEP % 2 == 0: #if it is even, then set the camera data to the model
        camera_data_batch = torch.stack([
            transform_camera(Image.fromarray(camera_data.squeeze().astype(np.uint8))) for camera_data in data_batch_camera[5]]).to(device)        
    
        # Set the camera data to the model
        features = model_camera(camera_data_batch)
        print("features.shape:", features.shape)
        #pass the features to the encoder
        encoded_features = model_encoder(features)
        print("encoded_features.shape:", encoded_features.shape)
        #print the encoded features tensor
        print(encoded_features)
        concatenated_all_camera = torch.cat((features, encoded_features, state_data_batch_camera), 1).to(device)
        print("CAMERA_concatenated_features.shape:", concatenated_all_camera.shape)
        print(concatenated_all_camera[0].dtype)
        print(model_actor)
        #print the features of the actor model
        print(model_actor(concatenated_all_camera))

        loss_mse_camera = F.mse_loss(model_actor(concatenated_all_camera), action_data_batch_camera)
        writer.add_scalar('MSE_LOSS_CAMERA/TRAIN', loss_mse_camera, TRAIN_STEP)



        model_camera_optimizer.zero_grad(set_to_none=True)
        model_encoder_optimizer.zero_grad(set_to_none=True)
        model_actor_optimizer.zero_grad(set_to_none=True)
        
        # Optimize and step *****************************************
        loss_mse_camera.backward()
        
        model_camera_optimizer.step()
        model_encoder_optimizer.step()
        model_actor_optimizer.step()
        
        model_scan_scheduler.step()
        model_encoder_scheduler.step()
        model_actor_scheduler.step()

        for param_group in model_camera_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)
        for param_group in model_encoder_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)
        for param_group in model_actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)

    else: #if it is odd, then set the scan data to the model
        scan_data_batch = torch.stack([
            torch.tensor(scan_data.reshape(1, -1), dtype=torch.float32) for scan_data in data_batch_scan[6]]).to(device)

        #normalize the scan data batch
        scan_data_batch = normalize_scan_data(scan_data_batch)
        print("scan_data_batch.shape:", scan_data_batch.shape)
        print("scan_data_batch data", scan_data_batch)

        # Set the scan data to the model
        features = model_scan(scan_data_batch)
        print("features.shape:", features.shape)
        #pass the features to the encoder
        encoded_features = model_encoder(features)
        print("encoded_features.shape:", encoded_features.shape)
        #print the encoded features tensor
        print(encoded_features)
        concatenated_all_scan = torch.cat((features, encoded_features, state_data_batch_scan), 1).to(device)
        print("SCAN_concatenated_features.shape:", concatenated_all_scan.shape)
        print(concatenated_all_scan[0].dtype)
        print(model_actor)
        #print the features of the actor model
        print(model_actor(concatenated_all_scan))
        #calculate the loss usning F.mse_loss by comparing the output from the actor model and the action data batch
        loss_mse_scan = F.mse_loss(model_actor(concatenated_all_scan), action_data_batch_scan)
        writer.add_scalar('MSE_LOSS_SCAN/TRAIN', loss_mse_scan, TRAIN_STEP)
        
        model_scan_optimizer.zero_grad(set_to_none=True)
        model_encoder_optimizer.zero_grad(set_to_none=True)
        model_actor_optimizer.zero_grad(set_to_none=True)
        
        # Optimize and step *****************************************
        loss_mse_scan.backward()
        
        model_scan_optimizer.step()
        model_encoder_optimizer.step()
        model_actor_optimizer.step()
        
        model_scan_scheduler.step()
        model_encoder_scheduler.step()
        model_actor_scheduler.step()

        for param_group in model_scan_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)
        for param_group in model_encoder_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)
        for param_group in model_actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)

    
    TRAIN_STEP += 1
    print("TRAIN_STEP:", TRAIN_STEP)

# Define the evaluation function
def evaluate(data_batch_camera, data_batch_scan):

    global EPOCH_STEP

    #convert the state data to tensor and set it to the device
    state_data_batch_camera = torch.tensor(data_batch_camera[0], dtype=torch.float32).to(device)
    print("action_data_batch_camera.shape:", state_data_batch_camera.shape)
    action_data_batch_camera = torch.tensor(data_batch_camera[1], dtype=torch.float32).to(device)
    print("action_data_batch_camera.shape:", action_data_batch_camera.shape)

    #convert the state data to tensor and set it to the device
    state_data_batch_scan = torch.tensor(data_batch_scan[0], dtype=torch.float32).to(device)
    print("state_data_batch_scan.shape:", state_data_batch_scan.shape)
    action_data_batch_scan = torch.tensor(data_batch_scan[1], dtype=torch.float32).to(device)
    print("state_data_batch_scan.shape:", state_data_batch_scan.shape)


    if EPOCH_STEP % 2 == 0: #if it is even, then set the camera data to the model
        with torch.no_grad():
            camera_data_batch = torch.stack([
                transform_camera(Image.fromarray(camera_data.squeeze().astype(np.uint8))) for camera_data in data_batch_camera[5]]).to(device)        

            # Set the camera data to the model
            features = model_camera(camera_data_batch)
            print("features.shape:", features.shape)
            #pass the features to the encoder
            encoded_features = model_encoder(features)
            print("encoded_features.shape:", encoded_features.shape)
            #print the encoded features tensor
            print(encoded_features)
            concatenated_all_camera = torch.cat((features, encoded_features, state_data_batch_camera), 1).to(device)
            print("CAMERA_concatenated_features.shape:", concatenated_all_camera.shape)
            print(concatenated_all_camera[0].dtype)
            print(model_actor)
            #print the features of the actor model
            print(model_actor(concatenated_all_camera))

            #calculate l2 loss
            loss_mse_camera = F.mse_loss(model_actor(concatenated_all_camera), action_data_batch_camera)
            loss_l1_camera = F.l1_loss(model_actor(concatenated_all_camera), action_data_batch_camera)
            writer.add_scalar('MSE_LOSS_CAMERA/EPOCH', loss_mse_camera, EPOCH_STEP)
            writer.add_scalar('L1_LOSS_CAMERA/EPOCH', loss_l1_camera, EPOCH_STEP)



    else: #if it is odd, then set the scan data to the model
        with torch.no_grad():
            scan_data_batch = torch.stack([
                torch.tensor(scan_data.reshape(1, -1), dtype=torch.float32) for scan_data in data_batch_scan[6]]).to(device)
            
            # Set the scan data to the model
            features = model_scan(scan_data_batch)
            print("features.shape:", features.shape)
            #pass the features to the encoder
            encoded_features = model_encoder(features)
            print("encoded_features.shape:", encoded_features.shape)
            #print the encoded features tensor
            print(encoded_features)
            concatenated_all_scan = torch.cat((features, encoded_features, state_data_batch_scan), 1).to(device)
            print("SCAN_concatenated_features.shape:", concatenated_all_scan.shape)
            print(concatenated_all_scan[0].dtype)
            print(model_actor)
            #print the features of the actor model
            print(model_actor(concatenated_all_scan))
            #calculate the loss usning F.mse_loss by comparing the output from the actor model and the action data batch
            loss_mse_scan = F.mse_loss(model_actor(concatenated_all_scan), action_data_batch_scan)
            loss_l1_scan = F.l1_loss(model_actor(concatenated_all_scan), action_data_batch_scan )
            writer.add_scalar('MSE_LOSS_SCAN/EPOCH', loss_mse_scan, EPOCH_STEP)
            writer.add_scalar('L1_LOSS_SCAN/EPOCH', loss_l1_scan, EPOCH_STEP)


    EPOCH_STEP += 1 
           

############## START: LOOP FOR TRAINING & EVALUATION THE MODEL ##############
iter = train_buffer.size()//BATCH_SIZE
print("iter:", iter)
epochs = 10000
for epoch in range(epochs):
    #print progress percentage the epoch number and delete the previous line
    print("Progress: %d%%, Epoch: %d" % ((epoch/epochs)*100, epoch), end="\r") 
    ################## START: CALL THE TRAIN FUNCTION FOR THE MODEL ##################      
    for i in range(iter):
        train(sample_batch_train_camera, sample_batch_train_scan)
    
    ################## END: CALL THE TRAIN FUNCTION FOR THE MODEL ####################
    
    ################## START: CALL THE EVALUATION FUNCTION FOR THE MODEL ##############
    if epoch > 0:
        evaluate(sample_batch_test_camera, sample_batch_test_scan)
        if epoch % 1 == 0:
            ename = "epoch %d.dat" % (epoch)
            fename = os.path.join(save_path, ename)
            model_camera.save_model(fename)
            model_scan.save_model(fename)
            model_encoder.save_model(fename)
            model_actor.save_model(fename)


        
    ################## END: CALL THE EVALUATION FUNCTION FOR THE MODEL ##################

ename = "epoch %d.dat" % (epoch)
fename = os.path.join(save_path, ename)
model_camera.save_model(fename)
model_scan.save_model(fename)
model_encoder.save_model(fename)
model_actor.save_model(fename)    
    
    
############## END: LOOP FOR TRAINING & EVALUATION THE MODEL ##############
