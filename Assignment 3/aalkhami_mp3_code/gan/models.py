import torch
from .spectral_normalization import SpectralNorm
import torch.nn as nn
class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(input_channels, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(1024, 2048, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # for part 1 in the assignment
        #self.fc = nn.Linear(1024, 1)
        # for part 2 in the assignment
        self.fc = nn.Linear(2048, 1)
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # return self.layers(x).squeeze().view(-1, 1)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

       
        ##########       END      ##########


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.noise_dim = noise_dim
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 2048, 4, 1, 0),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            #for part 2
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            #for part 1
            #nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # Reshape the noise tensor to (batch_size, noise_dim, 1, 1)
        x = x.view(x.size(0), self.noise_dim, 1, 1)
        return self.layers(x)
        
        ##########       END      ##########
    

