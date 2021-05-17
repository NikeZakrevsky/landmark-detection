import math

import numpy as np
import torch
from torchvision import transforms


class LandmarkDetector:

    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.expand_ratio = 0.2

    def _expand_box_ratio(self, image_width, image_height, box):
        face_ex_w, face_ex_h = (box[3] - box[2]) * self.expand_ratio, (box[1] - box[0]) * self.expand_ratio
        y1, x1 = int(max(math.floor(box[2] - face_ex_w), 0)), int(max(math.floor(box[0] - face_ex_h), 0))
        y2, x2 = int(min(math.ceil(box[3] + face_ex_w), image_width)), \
                 int(min(math.ceil(box[1] + face_ex_h), image_height))

        return x1, x2, y1, y2

    def _restore_origin_points(self, box, batch_locs, batch_scos, original_h, original_w):
        cropped_size = np.array([box[1] - box[0], box[3] - box[2], box[2], box[0]])

        cpu = torch.device('cpu')
        np_batch_locs, np_batch_scos = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy()
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)
        scale_h, scale_w = cropped_size[0] * 1. / original_h, cropped_size[1] * 1. / original_w
        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], \
                                           locations[:, 1] * scale_h + cropped_size[3]
        return np.concatenate((locations, scores), axis=1)

    def preprocess_image(self, image, box):
        expanded_box = self._expand_box_ratio(image.shape[1], image.shape[0], box)
        image_crop = image[expanded_box[0]:expanded_box[1], expanded_box[2]:expanded_box[3], :]
        return {'image': self.eval_transform(image_crop), 'box': expanded_box}

    def predict(self, image_info: dict):
        face_image = image_info['image']
        box = image_info['box']

        if face_image is None or face_image.size() == 0:
            return {
                'landmarks': [],
                'error_message': 'Empty image'
            }

        with torch.no_grad():
            inputs = face_image.unsqueeze(0)
            batch_heatmaps, batch_locs, batch_scos, _ = self.model(inputs)

        if batch_locs is None or batch_locs.size() == 0:
            return {
                'landmarks': [],
                'error_message': 'Landmarks not found'
            }

        return {
            'landmarks': self._restore_origin_points(box, batch_locs, batch_scos, inputs.size(-2), inputs.size(-1)),
            'error_message': ''
        }



