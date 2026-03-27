import torch
import torchvision.models as models
# Download the standard ResNet50 model
model = models.resnet50(weights='DEFAULT')
# Save it as the filename the app expects
torch.save(model.state_dict(), "ResNet50.pt")
print("Successfully created ResNet50.pt!")
