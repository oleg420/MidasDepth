import argparse

import torch
import torchvision

import cv2
import numpy as np

from PIL import Image

from MidasDepth import MidasDepth


def arg2source(x):
    try:
        return int(x)
    except:
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-em', '--encode_model', type=str, required=True)
    parser.add_argument('-s', '--source', type=arg2source, required=True)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MidasDepth(args.model, args.encode_model, device=device)

    cap = cv2.VideoCapture(args.source)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()

        depth = model(frame)

        cv2.imshow('Image', frame)
        cv2.imshow('Depth', np.array(depth))
        if cv2.waitKey(1) == 113:
            break

# from midas_.midas_net import MidasNet
#
# model = MidasNet('/home/oleg/.cache/torch/hub/checkpoints/model-f46da743.pt')
# model = model.cuda().eval()
#
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
#
# while True:
#     with torch.no_grad():
#         ret, frame = cap.read()
#
#         img = frame.copy()
#         img = Image.fromarray(img).convert('RGB')
#         img = torchvision.transforms.Resize((384, 384))(img)
#         img = torchvision.transforms.ToTensor()(img)
#         img = torchvision.transforms.Normalize(mean=mean, std=std)(img)
#         img = img.cuda()
#
#         prediction = model(img.unsqueeze(0))[0]
#
#         min = torch.min(prediction)
#         max = torch.max(prediction)
#         prediction = (prediction - min) / (max - min)
#
#         prediction = torchvision.transforms.ToPILImage()(prediction.detach().cpu())
#         prediction = torchvision.transforms.Resize(frame.shape[:2])(prediction)
#
#         cv2.imshow('frame', frame)
#         cv2.imshow('prediction', np.array(prediction))
#
#         if cv2.waitKey(1) == 113:
#             break