import torch

# Load traced model
traced_model = torch.jit.load("model_traced.pt")
traced_model.eval()

# Generate a face
z = torch.randn(1, 512)
with torch.no_grad():
    face = traced_model(z)
    # Denormalize from [-1, 1] to [0, 1]
    face = (face + 1) / 2
    face = torch.clamp(face, 0, 1)

# Save or display
from torchvision.utils import save_image
save_image(face, "generated_face.png")