'''
Created on 25.08.25
@author Rosie Eade

MACHINE LEARNING MODEL FUNCTIONS
and Multi-Linear Regression Functions.
For code to train ML model architectures using pytorch, e.g. 1D CNN

e.g. 
python rsdsMain_ObsPrediction.py

'''

import pandas as pd
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import calendar
import pdb

# ----------------------------------------------------------------------------
# MACHINE LEARNING MODEL FUNCTIONS - Set Up
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def get_ml_model(ml_model_name):
    """
    Function to get specific ML model architecture
    ML model codes:
    10c: Encoder MaxPool1d and Conv1d; Decoder ConvTranspose1d (concatenate so increase number of channels); Sigmoid layer
    """

    if ml_model_name=='10cS': model_ml=SequencePredictionCNN_10cSig()

    return model_ml

# ----------------------------------------------------------------------------
def get_loss_weights(wlist_txt):
    if wlist_txt=='W1000': weight_list = [1.0,0.0,0.0,0.0,np.arange(16)+17]
    if wlist_txt=='W0100': weight_list = [0.0,1.0,0.0,0.0,np.arange(16)+17]
    if wlist_txt=='W0010': weight_list = [0.0,0.0,1.0,0.0,np.arange(16)+17]
    if wlist_txt=='W0001': weight_list = [0.0,0.0,0.0,1.0,np.arange(16)+17]
    
    if wlist_txt=='W5500': weight_list = [0.5,0.5,0.0,0.0,np.arange(16)+17]
    if wlist_txt=='W5050': weight_list = [0.5,0.0,0.5,0.0,np.arange(16)+17]
    if wlist_txt=='W5005': weight_list = [0.5,0.0,0.0,0.5,np.arange(16)+17]
    if wlist_txt=='W0550': weight_list = [0.0,0.5,0.5,0.0,np.arange(16)+17]
    if wlist_txt=='W0505': weight_list = [0.0,0.5,0.0,0.5,np.arange(16)+17]
    if wlist_txt=='W0055': weight_list = [0.0,0.0,0.5,0.5,np.arange(16)+17]
    
    if wlist_txt=='W3330': weight_list = [0.3334,0.3333,0.3333,0.0,np.arange(16)+17]
    if wlist_txt=='W3303': weight_list = [0.3334,0.3333,0.0,0.3333,np.arange(16)+17]
    if wlist_txt=='W3033': weight_list = [0.3334,0.0,0.3333,0.3333,np.arange(16)+17]
    if wlist_txt=='W0333': weight_list = [0.0,0.3334,0.3333,0.3333,np.arange(16)+17]
    
    if wlist_txt=='W2222': weight_list = [0.25,0.25,0.25,0.25,np.arange(16)+17]

    return weight_list

# ----------------------------------------------------------------------------
# Custom Loss function
class ClimStatsLoss(nn.Module):
    def __init__(self, weight1=1, weight2=0.0, weight3=0.0, weight4=0.0, daylight_index=np.arange(48)):
        """
        Custom loss function as weighted average of multiple statistics:
	* 1. MSE of all sub-daily values
	* 2. Daily mean for daylight hours (MSE)
	* 3. Daily STD for daylight hours (MSE)
	* 4. Climatology for all sub-daily values in daylight hours (MSE)
        
        Args:
            weight1, weight2,...: Weights for different parts of loss function
            daylight_index: np array containing index of 30min timesteps to consider as daylight
        """
	
        super(ClimStatsLoss, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.weight4 = weight4
        self.daylight_index = daylight_index

        # Ensure weights sum to 1.0
        total_weight = weight1 + weight2 + weight3 + weight4

        # Validate weights
        if total_weight > 1:
            raise ValueError("Sum of weights must be = 1.0")

        self.mse = nn.MSELoss()
    
    def forward(self, outputs, targets):
        batch_size, seq_len = outputs.size()

        # Calculate point-wise MSE as a baseline
        loss1 = self.mse(outputs, targets)
        loss2 = self.mse(outputs[:,self.daylight_index].mean(axis=1), targets[:,self.daylight_index].mean(axis=1))
        loss3 = self.mse(outputs[:,self.daylight_index].std(axis=1), targets[:,self.daylight_index].std(axis=1))
        loss4 = self.mse(outputs[:,self.daylight_index].mean(axis=0), targets[:,self.daylight_index].mean(axis=0))
        # Combine losses with weights
        total_loss = self.weight1*loss1 + self.weight2*loss2 + self.weight3*loss3 + self.weight4*loss4
        
        return total_loss


# ----------------------------------------------------------------------------
# Training function
def train_model(model, train_loader, valid_loader, epochs=10, lr=0.001, Wlist=[1.0,0.0,0.0,0.0,np.arange(16)+17]):
    """
    Generic training structure for any given ML model architecture
    Options to adjust number of epochs and the learning rate (lr)
    Loss function based on the equally weighted sum of MSE Loss for Total and Direct irradiance
    Outputs loss for each term separately and combined for all epochs
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize the custom loss function and optimizer
    #criterion = nn.MSELoss()
    criterion = ClimStatsLoss(weight1=Wlist[0], weight2=Wlist[1], weight3=Wlist[2], weight4=Wlist[3], daylight_index=Wlist[4])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Defining relative weighting of loss function for Total vs Direct solar radiation
    weightTOT=0.5
    weightDIR=1.0 - weightTOT

    print('len(train_loader)')
    print(len(train_loader))
    print('len(valid_loader)')
    print(len(valid_loader))
    
    # Store complete and partial elements of loss function for each epoch
    #   TOT+DIR Training Loss; TOT+DIR Validation Loss; TOT Training Loss; TOT Validation Loss; DIR Training Loss; DIR Validation Loss
    loss_out=np.zeros([6,epochs])
    for epoch in range(epochs):
        # Training phase
        model.train()

        train_loss = 0.0
        trainTOT_loss = 0.0
        trainDIR_loss = 0.0
        for i, (short_seq, long_seq, target_seq, targetDIR_seq) in enumerate(train_loader):
            short_seq, long_seq, target_seq, targetDIR_seq = short_seq.to(device), long_seq.to(device), target_seq.to(device), targetDIR_seq.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, outputsDIR = model(short_seq, long_seq)
            lossTOT = criterion(outputs, target_seq)
            lossDIR = criterion(outputsDIR, targetDIR_seq)
            loss = weightTOT*lossTOT + weightDIR*lossDIR # Custom loss function, adding ok as autograd understands how to implement the backwards function
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            trainTOT_loss += lossTOT.item()
            trainDIR_loss += lossDIR.item()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        validTOT_loss = 0.0
        validDIR_loss = 0.0
        with torch.no_grad():
            for x_short, x_long, targets, targetsDIR in valid_loader:
                x_short, x_long, targets, targetsDIR = x_short.to(device), x_long.to(device), targets.to(device), targetsDIR.to(device)

                outputs, outputsDIR = model(x_short, x_long)
                lossTOT = criterion(outputs, targets)
                lossDIR = criterion(outputsDIR, targetsDIR)
                loss = weightTOT*lossTOT + weightDIR*lossDIR
                valid_loss += loss.item()
                validTOT_loss += lossTOT.item()
                validDIR_loss += lossDIR.item()
        
        # Print statistics
        # divide by length so have average loss rather than total
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Valid Loss: {valid_loss/len(valid_loader):.4f}')
        loss_out[0,epoch]=train_loss/len(train_loader)
        loss_out[1,epoch]=valid_loss/len(valid_loader)
        loss_out[2,epoch]=trainTOT_loss/len(train_loader)
        loss_out[3,epoch]=validTOT_loss/len(valid_loader)
        loss_out[4,epoch]=trainDIR_loss/len(train_loader)
        loss_out[5,epoch]=validDIR_loss/len(valid_loader)

    return model, loss_out
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# MACHINE LEARNING MODEL FUNCTIONS - A library of different architectures
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
class SequencePredictionCNN_10cSig(nn.Module):
    """
    Machine learning model
    Converts input 3 hour Allsky Total solar irradiance and 30min Clearsky Direct solar irradiance
    to
    Target 30min Allsky Total and Allsky Direct solar irradiance
    Input and Target have been pre-normalised as ratios relative to climatology statistics
    
    Encoder: MaxPool1d and Conv1d with no padding (long k5; short k3)
    Decoder: ConvTranspose1d
             + Sigmoid layer at the end for both Total and Direct so ratio in [0,1]
    """

    def __init__(self):
        super(SequencePredictionCNN_10cSig, self).__init__()
        
        # Process the sequence of length 8
        self.encoder_short = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
            # Output shape: [batch_size, 64, 6]
        )
        
        # Process the sequence of length 48
        self.encoder_long = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=0),
            # Output shape: [batch_size, 16, 44]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Output shape: [batch_size, 16, 22]

            nn.Conv1d(16, 32, kernel_size=3, padding=0),
            # Output shape: [batch_size, 32, 20]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Output shape: [batch_size, 32, 10]

            nn.Conv1d(32, 64, kernel_size=5, padding=0),
            # Output shape: [batch_size, 64, 6]
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # Combine the features and predict the output sequence
        # using reverse CNN
        self.decoderTOT = nn.Sequential(
            # Input shape: [batch_size, 64*2, 6]
            # Reshape layer will be handled in forward method
            
            # First upconv block: 6 -> 12
            nn.ConvTranspose1d(64*2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Output shape: [batch_size, 64, 12]
            
            # Second upconv block: 12 -> 24
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Output shape: [batch_size, 32, 24]
            
            # Third upconv block: 24 -> 48
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Output shape: [batch_size, 16, 48]
            
            # Final conv layer to map to output back to required size
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            # Output shape: [batch_size, 1, 48]
    
            nn.Sigmoid()  # Assuming normalized input data
            #nn.Tanh()  # Assuming normalized input data
            # Output shape: [batch_size, 1, 48]
        )
        self.decoderDIR = nn.Sequential(
            # Input shape: [batch_size, 64*2, 6]
            # Reshape layer will be handled in forward method
            
            # First upconv block: 6 -> 12
            nn.ConvTranspose1d(64*2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Output shape: [batch_size, 64, 12]
            
            # Second upconv block: 12 -> 24
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Output shape: [batch_size, 32, 24]
            
            # Third upconv block: 24 -> 48
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Output shape: [batch_size, 16, 48]
            
            # Final conv layer to map to output back to required size
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            # Output shape: [batch_size, 1, 48]
    
            nn.Sigmoid()  # Assuming normalized input data
            #nn.Tanh()  # Assuming normalized input data
            # Output shape: [batch_size, 1, 48]
        )
        
    def forward(self, x_short, x_long):
        # Reshape inputs to [batch_size, channels, sequence_length]
        # For 1D convolution, we need channels as the second dimension
        x_short = x_short.view(x_short.size(0), 1, -1)  # [batch_size, 1, 8]
        x_long = x_long.view(x_long.size(0), 1, -1)    # [batch_size, 1, 48]
        
        # Process each sequence
        short_features = self.encoder_short(x_short)  # [batch_size, 64, 6]
        long_features = self.encoder_long(x_long)     # [batch_size, 64, 6]
        
        # Concatenate features along the channel dimension
        combined_features = torch.cat((short_features, long_features), dim=1)  # [batch_size, 64*2, 6]

        # Generate the prediction
        outputTOT = self.decoderTOT(combined_features)  # [batch_size, 1, 48]
        outputDIR = self.decoderDIR(combined_features)  # [batch_size, 1, 48]
        
        # Reshape to the desired output format [batch_size, 48]
        outputTOT = outputTOT.squeeze(1)
        outputDIR = outputDIR.squeeze(1)
        
        return outputTOT, outputDIR

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# MULTI-LINEAR REGRESSION FUNCTION
# ----------------------------------------------------------------------------
def Tensor_least_squares_regression(X, y):
    """
    Compute multi-linear regression coefficients with least-squares
    
    Parameters:
    -----------
    X : torch.Tensor 
        e.g. shape (n_samples, 8)
    y : torch.Tensor 
        e.g. shape (n_samples, 48)

    Returns:
    --------
    beta: torch.Tensor
        Regression coefficients e.g. shape (8, 48)

    Uses Moore-Penrose pseudo-inverse incase an input channel has all identical values (constant values)
    β = X^+ y where X^+ is the pseudo-inverse of X
    """

    X_pinv = torch.linalg.pinv(X)
    beta = torch.matmul(X_pinv, y)

    return beta

# ----------------------------------------------------------------------------
def Tensor_LSR_predict(X, beta):
    """
    Compute predictions from given multi-linear regression coefficients
    
    Parameters:
    -----------
    X : torch.Tensor 
        Input predictors e.g. shape (n_samples, 8)
    beta : torch.Tensor
        Regression coefficients e.g. shape (8, 48)
    
    Returns:
    --------
    y_hat : torch.Tensor
        Predictions e.g. shape (n_samples, 48)
    """
    
    y_hat = torch.matmul(X, beta)
    
    return y_hat
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
