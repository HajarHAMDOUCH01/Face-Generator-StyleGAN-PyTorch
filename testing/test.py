import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision import transforms

import sys 

sys.path.append("/content/Convolutiional_VAE")
from vae_model import ConvolutionnalVAE

# checkpoint = torch.load('/content/vae_checkpoint_epoch_100.pth', map_location="cpu") 
z_dim = 256  
model = ConvolutionnalVAE(z_dim=z_dim, input_size=256) 
model = torch.jit.load("model_scripted.pt", map_location="cpu")
model.eval()  
# image_face1 = "/content/face1.jpg"
# image_face2 = "/content/test_face.jpg"

# face1 = Image.open(image_face1).convert("RGB")
# face2 = Image.open(image_face2).convert("RGB")

def generate_new_face(z_dim, model):
    with torch.no_grad():

        z_random = torch.randn(1, z_dim)  
        generated_face = model.decode(z_random)
        
        z_normal = torch.randn(1, z_dim) * 0.5  
        generated_face_normal = model.decode(z_normal)

    to_pil = ToPILImage()
    image = to_pil(generated_face_normal.squeeze(0))  

    image.save('generated_face_pil.png')

def interpolate_2_faces(model, face1, face2, alpha=0.5):
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        face1_tensor = transform(face1).unsqueeze(0)  # [1, 3, 256, 256]
        face2_tensor = transform(face2).unsqueeze(0)

        # Encode
        _, mu1 , logvar1 = model.encode(face1_tensor)
        _, mu2 , logvar2 = model.encode(face2_tensor)

        z1 = model.reparametrize(mu1, logvar1)
        z2 = model.reparametrize(mu2, logvar2)

        # Interpolate
        z_seed = z1 * (1 - alpha) + z2 * alpha
        z_seed = z_seed.unsqueeze(0) if z_seed.ndim == 1 else z_seed

        # Decode
        mixed_face = model.decode(z_seed)

    # Convert to PIL
    to_pil = ToPILImage()
    image = to_pil(mixed_face.squeeze(0).cpu().clamp(0,1))
    image.save("mixed_face.png")
    return image

if __name__=="__main__":
    # interpolate_2_faces(model, face1, face2, alpha=0.5)

    generate_new_face(256, model)
