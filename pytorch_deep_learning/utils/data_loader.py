import os
import io
from PIL import Image

import torch
from torchvision.transforms import ToTensor


class ShepardMetzler:
    def __init__(self, data_path, transformer):
        self.data_path = data_path
        self.transformer = transformer

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, file_name):
        byte_file_path = os.path.join(self.data_path, '{}.pt'.format(file_name))
        data = torch.load(byte_file_path)

        to_tensor = lambda byte_image: ToTensor()(Image.open(io.BytesIO(byte_image)))
        images = torch.stack([to_tensor(frame) for frame in data.frames])

        viewpoints = torch.from_numpy(data.camera)
        viewpoints = viewpoints(-1, 5)

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            viewpoints =self.target_transform(viewpoints)

        return images, viewpoints
