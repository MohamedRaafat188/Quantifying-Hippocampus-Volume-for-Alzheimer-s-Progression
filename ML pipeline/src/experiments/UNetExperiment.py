"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_prep.VolumesDataset import VolumesDataset
from utils.volume_stats import Dice3d
from networks.UNets3D import UNet3D, ResidualUNet3D
from inference.UNetInferenceAgent import UNetInferenceAgent
from lib.losses3D.dice import DiceLoss


class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task.
    The basic life cycle of a UNetExperiment is:

        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.min_val_loss = np.Inf
        self.max_dice_score = -1
        self.name = config.name
        self.train_losses = []
        self.valid_losses = []
        self.valid_dc = []

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create data loaders
        self.train_loader = DataLoader(VolumesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(VolumesDataset(dataset[split["val"]]),
                    batch_size=config.batch_size, shuffle=True, num_workers=0)
        

        # we will access volumes directly for testing
        self.test_data = dataset[split["test"]]

        # Do we have CUDA available?
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure our model and other training implements
        if config.name == "ResidualUNet3D":
            self.model = ResidualUNet3D(in_channels=1, out_channels=3, f_maps=config.features_maps, final_sigmoid=False)
        elif config.name == "UNet3D":
            self.model = UNet3D(in_channels=1, out_channels=3, final_sigmoid=False)

        self.model.to(self.device)

        # We are using a combination of standard cross-entropy loss since the model output is essentially
        # a tensor with softmax'd prediction of each pixel's probability of belonging 
        # to a certain class and a Dice loss because it is more suitable for segmentation problems
        self.loss_function1 = DiceLoss(classes=3, sigmoid_normalization=False)
        self.loss_function2 = torch.nn.CrossEntropyLoss()

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Scheduler helps us update learning rate automatically
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        print(f"Epoch {self.epoch+1}...")
        self.model.train()
        loss_list = []

        # Loop over our minibatches
        for batch in self.train_loader:
            self.optimizer.zero_grad()

            data = batch['image'].to(self.device)
            target = batch['seg'].to(self.device)

            prediction = self.model(data).to('cpu')

            target = target.type(torch.LongTensor).to('cpu')

            loss1, _ = self.loss_function1(prediction, target)
            loss2 = self.loss_function2(prediction, target)
            loss = loss1 + loss2

            loss_list.append(loss.item())

            loss.backward()
            self.optimizer.step()

        print(f"Mean Training Loss = {np.mean(loss_list)}")
        self.train_losses.append(np.mean(loss_list))


    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []
        dc_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                
                data = batch['image'].to(self.device)
                target = batch['seg'].to(self.device)

                prediction = self.model(data)
               
                prediction = self.model(data).to('cpu')
                target = target.type(torch.LongTensor).to('cpu')

                loss1, _ = self.loss_function1(prediction, target)
                loss2 = self.loss_function2(prediction, target)
                loss = loss1 + loss2
                loss_list.append(loss.item())

                prediction = torch.argmax(prediction, dim=1)

                dice_score = Dice3d(prediction, target)
                dc_list.append(dice_score[0])
                
        mean_val_loss = np.mean(loss_list)
        mean_dice_score = np.mean(dc_list)

        print(f"Mean Validation Loss = {mean_val_loss}")

        if mean_val_loss < self.min_val_loss:
            self.save_model_parameters()
            print("Val loss decreased. Model updated.")
            self.min_val_loss = mean_val_loss
        print()

        self.valid_dc.append(mean_dice_score)
        self.valid_losses.append(mean_val_loss)


    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")

        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        print("Testing...")
        self.model.eval()

        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes, much like we will be doing it when we deploy the model in the 
        # clinical environment. 

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []

        # for every in test set
        for x in self.test_data:
            pred_label = inference_agent.single_volume_inference(x["image"])

            # We compute and report Dice similarity coefficient which 
            # assess how close our volumes are to each other

            dc = Dice3d(pred_label, x["seg"])
            dc_list.append(dc)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc
                })
            
        out_dict["overall"] = {
            "mean_dice1": np.mean([dsc[0] for dsc in dc_list]),
            "mean_dice2": np.mean([dsc[1] for dsc in dc_list]),
            "mean_obj_dice": np.mean([dsc[2] for dsc in dc_list])}

        print("Mean DSC1 = {}".format(out_dict["overall"]["mean_dice1"]))
        print("Mean DSC2 = {}".format(out_dict["overall"]["mean_dice2"]))
        print("Mean OBJ DSC = {}".format(out_dict["overall"]["mean_obj_dice"]))

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()
        return self.train_losses, self.valid_losses, self.valid_dc
