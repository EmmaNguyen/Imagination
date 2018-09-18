import os
import io
import collections
from PIL import Image

import torch
from torchvision.transforms import ToTensor

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

# Difference: Context, Scence, Image and Frame

def vertical_tranverse_rotation(viewpoint):
    """
    Transforms the viewpoint vector into a consistent
    representation, a.k.a mental rotation in two direction
    """
    # viewpoint ~ position, [yaw, pitch]
    position, principle_axes = torch.split(viewpoint, 3, dim=-1)
    yaw, pitch = torch.split(principle_axes, 1, dim=-1)

    viewvector = [position, torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)]
    viewpoint_hat = torch.cat(viewvector, dim=-1)

    return viewpoint_hat

class ShepardMetzler:

    def __init__(self, data_path, transformer=None, target_transformer=None):
        self.data_path = data_path
        self.transformer = transformer
        self.target_transformer = target_transformer

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, file_name):
        byte_file_path = os.path.join(self.data_path, '{}.pt'.format(file_name))
        # print(byte_file_path)
        # import pdb; pdb.set_trace()
        data = torch.load(byte_file_path)

        to_tensor = lambda byte_image: ToTensor()(Image.open(io.BytesIO(byte_image)))
        images = torch.stack([to_tensor(frame) for frame in data.frames])

        viewpoints = torch.from_numpy(data.cameras)
        viewpoints = viewpoints.view(-1, 5)  #Hard code?

        if self.transformer:
            images = self.transformer(images)

        if self.target_transformer:
            viewpoints =self.target_transformer(viewpoints)

        return images, viewpoints
