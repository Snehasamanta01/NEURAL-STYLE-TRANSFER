import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# =============================
# Device configuration
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Image size
# =============================
image_size = 512 if torch.cuda.is_available() else 256

# =============================
# Image loader
# =============================
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Load images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

assert content_img.size() == style_img.size(), "Images must be same size"

# =============================
# Display function
# =============================
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")

# =============================
# Content Loss
# =============================
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# =============================
# Style Loss
# =============================
def gram_matrix(input):
    batch_size, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# =============================
# Normalization
# =============================
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# =============================
# Load pretrained VGG19
# =============================
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# =============================
# Layers
# =============================
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# =============================
# Build model
# =============================
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):

    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError("Unrecognized layer")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]
    return model, style_losses, content_losses

# =============================
# Input image
# =============================
input_img = content_img.clone()

# =============================
# Optimizer
# =============================
optimizer = optim.LBFGS([input_img.requires_grad_()])

# =============================
# Style Transfer
# =============================
def run_style_transfer(num_steps=300, style_weight=1000000, content_weight=1):
    print("Starting Style Transfer...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        style_img,
        content_img
    )

    step = [0]
    while step[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            input_img.data.clamp_(0, 1)

            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            step[0] += 1
            if step[0] % 50 == 0:
                print(f"Step {step[0]} | Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# =============================
# Run & Save
# =============================
output = run_style_transfer()
torchvision.transforms.ToPILImage()(output.squeeze(0)).save("output.png")

# =============================
# Show result
# =============================
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
imshow(content_img, "Content")
plt.subplot(1,3,2)
imshow(style_img, "Style")
plt.subplot(1,3,3)
imshow(output, "Output")
plt.show()
