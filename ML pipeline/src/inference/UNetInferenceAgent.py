"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.UNets3D import UNet3D


class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu"):

        self.model = model
        self.device = device

        if model is None:
            self.model = UNet3D(in_channels=1, out_channels=3, final_sigmoid=False)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        volume_tensor = torch.from_numpy(volume.astype(np.single)/np.max(volume)).unsqueeze(0).unsqueeze(0)
        pred = self.model(volume_tensor.to(self.device))
        pred = np.squeeze(pred.cpu().detach())
        pred = torch.argmax(pred, dim=0)
        return pred 
