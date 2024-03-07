import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import cv2
from distillation_CNNs import CNN_FeatureExtractor_Camera, CNN_FeatureExtractor_Scan, Encoder
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Initialize empty buffers for T1 and T2


BATCH_SIZE = 32

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

# Check sizes
print("Expected size of the training data:", int(replay_buffer.size() * 0.8))
print("Actual size of the training buffer:", train_buffer.size())
print("Expected size of the testing data:", replay_buffer.size() - int(replay_buffer.size() * 0.8))
print("Actual size of the testing buffer:", test_buffer.size())


# show a sample batch from the replay buffer
sample_batch_train = train_buffer.sample_batch(BATCH_SIZE)
sample_batch_test = test_buffer.sample_batch(BATCH_SIZE)     


#################PRINTING THE SAMPLE BATCH####################

# Show sample batch from the replay buffer with titles and labels
# print("Sample batch from the replay buffer")
# print("state:", sample_batch_train[0][1145])
# print("\naction:", sample_batch_train[1][1145])
# print("\nreward:", sample_batch_train[2])
# print("\ndone:", sample_batch_train[3])
# print("\nnext_state:", sample_batch_train[4])
# print("\ncamera_data_batch:", sample_batch_train[5])
# print("\nscan_data_batch:", sample_batch_train[6])
#################PRINTING THE SAMPLE BATCH####################

###############PLOTINMG THE IMAGES####################
# import cv2
# import numpy as np

# # Define grid size and grid image size
# grid_size = (8, 4)
# slot_height, slot_width = 32, 64  # Size of each slot in the grid
# grid_image = np.zeros((slot_height * grid_size[0], slot_width * grid_size[1]), dtype=np.uint8)

# # Iterate over the grid positions
# for i in range(grid_size[0]):
#     for j in range(grid_size[1]):
#         # Extract the image from the batch, assuming it's the 5th sample batch and squeezing out any singleton dimensions
#         image = sample_batch_test[5][i * grid_size[1] + j].squeeze().astype(np.uint8)
        
#         # Resize the image to fit into the slot size (32x64)
#         resized_image = cv2.resize(image, (slot_width, slot_height))

#         # Place the resized image into the grid
#         grid_image[i * slot_height: (i + 1) * slot_height, j * slot_width: (j + 1) * slot_width] = resized_image

# # Create a named window that can be resized
# cv2.namedWindow("Grid of Images", cv2.WINDOW_NORMAL)

# # Resize the window, for example to 1024x768 pixels or any other size you prefer
# cv2.resizeWindow("Grid of Images", 1024, 768)

# # Show the grid of images
# cv2.imshow("Grid of Images", grid_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
###############PLOTINMG THE IMAGES####################

transform_camera = Compose([
    Resize((32, 64)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# transofrm_scan = Compose([
#     ToTensor(),
#     Normalize(mean=[0.5], std=[0.5])
# ])

camera_data_batch = torch.stack([
    transform_camera(Image.fromarray(camera_data.squeeze().astype(np.uint8))) for camera_data in sample_batch_train[5]
])

scan_data_batch = torch.stack([
    torch.tensor(scan_data.reshape(1, -1), dtype=torch.float32) for scan_data in sample_batch_train[6]
])

print("Scan data batch shape:", scan_data_batch.shape) #Scan data batch shape: torch.Size([32, 1, 360])

print("Camera data batch shape:", camera_data_batch.shape) #Camera data batch shape: torch.Size([32, 1, 32, 64])

OUTPUT_DIM = 10

# ASK MURAD ABOUT THE EVAL MODE OR TRAIN MODE
model_camera = CNN_FeatureExtractor_Camera().eval().to(device)
model_scan = CNN_FeatureExtractor_Scan().eval().to(device)
model_encoder = Encoder(OUTPUT_DIM).eval().to(device)


# Define the train fucntion
def train(model_camera, model_scan, model_encoder, data_batch):
    # create a temporary variable to increase +1 after each iteration.
    # This varible will be used to set set either camera data or scan data to the model. 
    # if it is even, then set the camera data to the model, otherwise set the scan data to the model.
    temp = 0
    if temp % 2 == 0:
        print("Even Number - Camera data:", data_batch[5][i])
        
    else:
        print("Odd Number - Scan data:", data_batch[6][i])

    temp += 1
        


# Define the evaluation function
def evaluate(model, data_batch, criterion):
    pass



############## START: LOOP FOR TRAINING & EVALUATION THE MODEL ##############
iter = train_buffer.size()//BATCH_SIZE
print("iter:", iter)
epochs = 10000
for epoch in range(epochs):
    #print progress percentage the epoch number and delete the previous line
    print("Progress: %d%%, Epoch: %d" % ((epoch/epochs)*100, epoch), end="\r") 
    ################## START: CALL THE TRAIN FUNCTION FOR THE MODEL ##################      
    for i in range(iter):
        train(model_camera, model_scan, model_encoder, sample_batch_train)
    
    ################## END: CALL THE TRAIN FUNCTION FOR THE MODEL ####################
    
    ################## START: CALL THE EVALUATION FUNCTION FOR THE MODEL ##############
    if epoch > 0:
        pass
        
    ################## END: CALL THE EVALUATION FUNCTION FOR THE MODEL ##################
    
############## END: LOOP FOR TRAINING & EVALUATION THE MODEL ##############