import torch.nn as nn
import torch
import os


class CNN_FeatureExtractor_Camera(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)  # Modified input channels to 1
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # add fully connected layer to reduce the number of features to 512
        self.fc1 = nn.Linear(32*16*32, 512)
        self.act3 = nn.ReLU() # Ask Murad


    def forward(self, x):
        # Apply the convolutional layers
        x = self.drop1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        
        # Flatten the feature map
        x = self.flatten(x)

        x = self.act3(self.fc1(x))
        
        # Return the flattened feature map
        return x
    
    def save_model(self, path):
    # Check if the directory exists
        if not os.path.exists(os.path.dirname(path)):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path))
        # Save the model
        torch.save(self.state_dict(), path + "camera_model")
        print("Camera Model saved successfully")
    
    # function to load the model
    def load_model(self, path):
        self.load_state_dict(torch.load(path) + "camera_model")
        print("Camera Model loaded successfully")


#create a CNN for the scan data
class CNN_FeatureExtractor_Scan(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)  # Modified input channels to 1
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # add fully connected layer to reduce the number of features to 512. The input size is based on the output of the flatten layer
        self.fc1 = nn.Linear(5760, 512)
        self.act3 = nn.ReLU()

        

    def forward(self, x):
        # Apply the convolutional layers
        x = self.drop1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        
        # Flatten the feature map
        x = self.flatten(x)

        x = self.act3(self.fc1(x))
        
        # Return the flattened feature map
        return x
    
    # function to save the model
    def save_model(self, path):
        torch.save(self.state_dict(), path+ "scan_model")
        print("Scan Model saved successfully")

    # function to load the model
    def load_model(self, path):
        self.load_state_dict(torch.load(path + "scan_model"))
        print("Scan Model loaded successfully")



class Encoder(nn.Module):
    def __init__(self, output_dim):  # Add output_dim as a parameter
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 400),  # Assuming 32*16*32 is the flattened size from the previous layer
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)  # Use output_dim here
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
    def save_model(self, path):
        torch.save(self.state_dict(), path + "encoder_model")
        print("Encoder Model saved successfully")

    # function to load the model
    def load_model(self, path):
        self.load_state_dict(torch.load(path) + "encoder_model")
        print("Encoder Model loaded successfully")

# create a actor model for the concatenated_all
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.actor(x)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path + "actor_model")
        print("Actor Model saved successfully")

    # define a function to load the model
    def load_model(self, path):
        self.load_state_dict(torch.load(path) + "actor_model")
        print("Actor Model loaded successfully")
