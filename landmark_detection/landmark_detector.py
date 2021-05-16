import torch
from torchvision import transforms
import numpy as np
import math


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
        face_ex_w, face_ex_h = (box[2] - box[0]) * self.expand_ratio, (box[3] - box[1]) * self.expand_ratio
        y1, x1 = int(max(math.floor(box[0] - face_ex_w), 0)), int(max(math.floor(box[1] - face_ex_h), 0))
        y2, x2 = int(min(math.ceil(box[2] + face_ex_w), image_width)), \
                 int(min(math.ceil(box[3] + face_ex_h), image_height))

        return y1, x1, y2, x2

    def _restore_origin_points_position(self, box, batch_locs, batch_scos, original_h, original_w):
        cropped_size = np.array([box[2] - box[0], box[3] - box[1], box[1], box[0]])

        cpu = torch.device('cpu')
        np_batch_locs, np_batch_scos = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy()
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(np_batch_scos[0, :-1], -1)
        scale_h, scale_w = cropped_size[0] * 1. / original_h, cropped_size[1] * 1. / original_w
        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], \
                                           locations[:, 1] * scale_h + cropped_size[3]
        return np.concatenate((locations, scores), axis=1).transpose(1, 0)

    def preprocess_image(self, image, box):
        image_crop = image[box[0]:box[2], box[1]:box[3], :]
        return self.eval_transform(image_crop)

    def predict(self, image_info: dict):
        face_image = image_info['image']
        box = image_info['box']

        with torch.no_grad():
            inputs = face_image.unsqueeze(0)
            batch_heatmaps, batch_locs, batch_scos, _ = self.model(inputs)

        self._restore_origin_points_position(box, batch_locs, batch_scos, inputs.size(-2), inputs.size(-1))

