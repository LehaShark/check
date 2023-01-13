from albumentations.pytorch import ToTensorV2
from flask import Flask
import torch
import os
import albumentations as A
import numpy as np
from PIL import Image

from nets import resnet50

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'weight_decay', '74.pth')

    device = torch.device('cuda')
    model = resnet50(pretrained=True, num_classes=1)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    im = np.asarray(Image.open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'train', 'check',
                     'check_with_background85.png')))

    def Predict_Image(image: np.ndarray = None):
        """
        classes:
        0) 'receipt'
        1) 'negative'
        """

        normalize = [A.Normalize(mean=[0.6518, 0.6298, 0.6251],
                                 std=[0.2657, 0.2781, 0.2840]),
                     ToTensorV2()
                     ]

        transforms = A.Compose([A.Resize(height=256, width=256),
                                A.CenterCrop(224, 224),
                                *normalize
                                ])

        image = transforms(image=image)['image']

        prediction = model(image[None, :])

        predict = (prediction > 0.5).long().cpu()

        if not predict:
            return True
        else:
            return False


    is_receipt = Predict_Image(im)
    j = 0





