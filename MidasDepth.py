import torch
import torchvision

from PIL import Image

from midas.midas_net import MidasNet


class MidasDepth:
    def __init__(self, model_path, encoder_path, device='cpu', min_dist=10, max_dist=10000):
        self.device = device

        self.model = MidasNet(model_path, encoder_path)
        self.model = self.model.to(self.device).eval()

        self.min_dist = min_dist
        self.max_dist = max_dist

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image):
        h, w, _ = image.shape
        img = Image.fromarray(image).convert('RGB')
        img = torchvision.transforms.Resize((384, 384))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(img)
        img = img.to(self.device)

        out = self.model(img.unsqueeze(0)).squeeze(0)

        out = (out - self.min_dist) / (self.max_dist - self.min_dist)
        out = torch.clamp(out, 0, 1)

        out = torchvision.transforms.ToPILImage()(out.detach().cpu())
        out = torchvision.transforms.Resize((h, w))(out)

        return out