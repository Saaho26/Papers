import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import torchvision.transforms as transforms 

# Add Noise to the image 
def diffusion_forwardpass(x_0 , t  = 3 , noise = None):
    T = 10000
    beta = torch.linspace(0.0001 , 0.2 , T )
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha , dim=  0)

    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
    one_minus_alpha = torch.sqrt(1 - alpha_bar[t])
    x_t = sqrt_alpha_bar * x_0 + one_minus_alpha * noise
    return x_t



# For visualize the image 
def visualize_image(image_data , timestep = [1, 10 , 100 , 1000]):
    plt.figure(figsize=(15  , 5))
    for i ,t in enumerate(timestep):
        noise_image = diffusion_forwardpass(image_data , t)
        noise_image = noise_image.squeeze().permute(1,2,0).detach().numpy()

        plt.subplot(1, len(timestep) , i+1)
        plt.imshow(noise_image)
        plt.title(f'Time Step: {t}')
    plt.show()


# Load the image data 
def load_image(image_path , transform  = None):
    image  =Image.open(image_path)
    if transform is None:
    
        transform  = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
    
    image_data = transform(image)
    return image_data
        
    

image = input("enter image ")
image = load_image(image)
visualize_image(image)

