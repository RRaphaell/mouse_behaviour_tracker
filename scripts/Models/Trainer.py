import gc
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader,
                 *, optimizer, loss_fn, dice_loss, iou_loss, config):

        self.model = model
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dice_loss = dice_loss
        self.iou_loss = iou_loss
        self.config = config
        self.scheduler = self.fetch_scheduler()

        # for logs
        self.logs = {"train_loss": [],
                     "valid_loss": [],
                     "valid_iou": [],
                     "valid_dice": [],
                     "lr": []}

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        dataset_size, running_loss, dice, iou = 0, 0.0, 0, 0

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='Valid ')
        for step, (images, masks) in pbar:
            images = images.to(self.config.device, dtype=torch.float)
            masks = masks.to(self.config.device, dtype=torch.float)
            batch_size = images.size(0)

            y_pred = self.model(images)
            loss = self.loss_fn(y_pred, masks)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size

            y_pred = torch.nn.Sigmoid()(y_pred)
            dice += self.dice_loss(masks, y_pred).cpu().detach().numpy()
            iou += self.iou_loss(masks, y_pred).cpu().detach().numpy()

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
            images = images.to(self.config.device, dtype=torch.float)
            masks = masks.to(self.config.device, dtype=torch.float)
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

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}', lr=f'{current_lr:0.5f}', gpu_mem=f'{mem:0.2f} GB')

        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss

    def run_training(self):
        if torch.cuda.is_available():
            print("cuda: {}\n".format(torch.cuda.get_device_name()))

        best_model = copy.deepcopy(self.model.state_dict())
        best_dice = -np.inf
        best_iou = -np.inf

        for epoch in range(1, self.config.n_epoch + 1):
            gc.collect()
            print(f'Epoch {epoch}/{self.config.n_epoch}')
            train_loss = self.train_one_epoch()
            val_loss, val_dice, val_iou = self.valid_one_epoch()

            # log values
            self.logs["train_loss"].append(train_loss)
            self.logs["valid_loss"].append(val_loss)
            self.logs["valid_iou"].append(val_iou)
            self.logs["valid_dice"].append(val_dice)
            self.logs["lr"].append(self.optimizer.param_groups[0]['lr'])

            if (val_dice >= best_dice) and (val_iou >= best_iou):
                print(
                    f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})  ({best_iou:0.4f} ---> {val_iou:0.4f})")
                best_dice = val_dice
                best_iou = val_iou
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), f"{self.config.model_out_dir}best_model.bin")
                print("Model Saved")

        print(f"Best score  dice: {best_dice:0.4f}  iou: {best_iou:0.4f}")
        self.model.load_state_dict(best_model)

    def fetch_scheduler(self):
        if self.config.scheduler == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                             T_max=self.config.T_max, eta_min=self.config.min_lr)
        elif self.config.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                       T_0=self.config.T_0, eta_min=self.config.min_lr)
        elif self.config.scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=7,
                                                             threshold=0.0001, min_lr=self.config.min_lr)
        elif self.config.scheduler == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.85)
        elif self.config.scheduler is None:
            return None

        return scheduler

    def plot_logs(self):
        num_epochs = len(self.logs["train_loss"])

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].plot(np.arange(1, num_epochs + 1), self.logs["train_loss"], label='Train loss')
        ax[0].plot(np.arange(1, num_epochs + 1), self.logs["valid_loss"], label='Valid loss')
        ax[0].legend()

        ax[1].plot(np.arange(1, num_epochs + 1), self.logs["valid_iou"], label='Valid Iou')
        ax[1].plot(np.arange(1, num_epochs + 1), self.logs["valid_dice"], label='Valid Dice')
        ax[1].legend()

        ax[2].plot(np.arange(1, num_epochs + 1), self.logs["lr"], label='Learning rate')
        ax[2].legend()

        plt.tight_layout()

    def show_result(self, dataset, num_examples):
        figure = plt.figure(figsize=(10, num_examples * 3))
        cols, rows = 3, num_examples

        for i in range(1, cols * rows + 1, 3):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]

            img = img.to(self.config.device, dtype=torch.float)
            img = torch.unsqueeze(img, 0)
            with torch.no_grad():
                pred = self.model(img)
                pred = (nn.Sigmoid()(pred) > 0.5).double()

            img = img.cpu().detach()
            pred = pred.cpu().detach()

            ax1 = figure.add_subplot(rows, cols, i)
            ax1.imshow(img[0][0])
            ax1.set_title("image")

            ax2 = figure.add_subplot(rows, cols, i + 1)
            ax2.imshow(torch.mean(label.float(), dim=0))
            ax2.set_title("ground truth")

            ax3 = figure.add_subplot(rows, cols, i + 2)
            ax3.imshow(torch.mean(pred[0], dim=0))
            ax3.set_title("predicted")

        plt.tight_layout()
