import gc
import copy
import numpy as np
from tqdm import tqdm

import torch


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader,
                 *, scheduler, optimizer, loss_fn, dice_loss, iou_loss, config):

        self.model = model
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dice_loss = dice_loss
        self.iou_loss = iou_loss
        self.device = config.device
        self.num_epochs = config.n_epoch
        self.model_out_dir = config.model_out_dir

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        dataset_size, running_loss = 0, 0.0

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='Valid ')
        for step, (images, masks) in pbar:
            images = images.to(self.device, dtype=torch.float)
            masks = masks.to(self.device, dtype=torch.float)
            batch_size = images.size(0)

            y_pred = self.model(images)
            loss = self.loss_fn(y_pred, masks)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size

            dice = self.dice_loss(masks, y_pred).cpu().detach().numpy()
            iou = self.iou_loss(masks, y_pred).cpu().detach().numpy()

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}', lr=f'{current_lr:0.5f}', dice=f'{dice:0.5f}',
                             iou=f'{iou:0.5f}', gpu_mem=f'{mem:0.2f} GB')

        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss, dice, iou

    def train_one_epoch(self):
        self.model.train()
        dataset_size, running_loss = 0, 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train ')
        for step, (images, masks) in pbar:
            images = images.to(self.device, dtype=torch.float)
            masks = masks.to(self.device, dtype=torch.float)
            batch_size = images.size(0)

            y_pred = self.model(images)
            loss = self.loss_fn(y_pred, masks)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size

            dice = self.dice_loss(masks, y_pred).cpu().detach().numpy()
            iou = self.iou_loss(masks, y_pred).cpu().detach().numpy()
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}', lr=f'{current_lr:0.5f}', dice=f'{dice:0.5f}',
                             iou=f'{iou:0.5f}', gpu_mem=f'{mem:0.2f} GB')

        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss

    def run_training(self):
        if torch.cuda.is_available():
            print("cuda: {}\n".format(torch.cuda.get_device_name()))

        best_model = copy.deepcopy(self.model.state_dict())
        best_dice = -np.inf
        best_iou = -np.inf
        best_epoch = -1

        for epoch in range(1, self.num_epochs + 1):
            gc.collect()
            print(f'Epoch {epoch}/{self.num_epochs}')
            train_loss = self.train_one_epoch()
            val_loss, val_dice, val_iou = self.valid_one_epoch()

            if (val_dice >= best_dice) and (val_iou >= best_iou):
                print(
                    f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})  ({best_iou:0.4f} ---> {val_iou:0.4f})")
                best_dice = val_dice
                best_iou = val_iou
                best_epoch = epoch
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), f"{self.model_out_dir}best_model.bin")
                print("Model Saved")

        print(f"Best score  dice: {best_dice:0.4f}  iou: {best_iou:0.4f}")
        self.model.load_state_dict(best_model)

        return self.model
