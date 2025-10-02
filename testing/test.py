import torch

"""traced model takes ~ 5 seconds on cpu to generate a face"""

# Load traced model
scripted_model = torch.jit.load("./scripted_model.pt")
scripted_model.eval()

# Generate a face
z = torch.randn(1, 512)
with torch.no_grad():
    face = scripted_model(z)
    # Denormalize from [-1, 1] to [0, 1]
    face = (face[0] + 1) / 2
    face = torch.clamp(face, 0, 1)

from torchvision.utils import save_image
save_image(face, "generated_face.png")