import numpy as np
import torch

from core.Trainer import Trainer
from time import time
import wandb
import logging
from optim.losses.image_losses import *
import matplotlib.pyplot as plt
import copy
from torch.nn import KLDivLoss, L1Loss
from optim.metrics.rl_metrics import *
from optim.metrics.rec_metrics import *
import pandas as pd

from dl_utils.vizu_utils import *

from model_zoo.beta_vae_higgings import initialize_weights
from torchmetrics.classification import Accuracy
from optim.losses.ln_losses import L2
import os
import json

import io
from PIL import Image


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        self.criterion_KLD = KLDivLoss().to(device)
        #self.num_classes = model.num_classes
        self.fctr = training_params['fctr']
        self.l1_crit = L1Loss(reduction="sum")

        self.loss_type = training_params['loss_type'] if 'loss_type' in training_params.keys() else 'mse'
        self.annealing = training_params['annealing'] if 'annealing' in training_params.keys() else 1

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer

        self.model.apply(initialize_weights)
        epoch_losses = []
        epoch_losses_pl = []
        epoch_losses_rec = []

        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue

            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break

            start_time = time()
            batch_loss, batch_loss_reg, batch_loss_rec,count_images = 1.0, 1.0, 1.0, 0
            batch_loss_pl = 1.0

            z_save = []
            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                labels = data[1].to(self.device)
                attributes = data[2].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape
                count_images += b

                # Forward Pass
                self.optimizer.zero_grad()
                reconstructed_images, f_result = self.model(transformed_images)

                # Reconstruction Loss
                loss = self.criterion_rec(reconstructed_images,transformed_images,f_result, labels, attributes)
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(transformed_images, reconstructed_images)
                    loss  = loss + self.annealing * pl_error
                else:
                    weight_reg_loss = 0
                    for param in self.model.parameters():
                        weight_reg_loss += self.l1_crit(param, target=torch.zeros_like(param))
                    loss += self.fctr * weight_reg_loss

                    pl_error = loss
                # Backward Pass
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # to avoid nan loss
                self.optimizer.step()
                batch_loss += loss.item() * images.size(0)
                batch_loss_pl += pl_error.item() * images.size(0)
                #batch_loss_rec += loss_rec.item() * images.size(0)
                z_save.append(f_result['z'])

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_reg
            #epoch_loss_rec = batch_loss_rec / count_images if count_images > 0 else batch_loss_rec

            epoch_losses.append(epoch_loss)
            epoch_losses_pl.append(epoch_loss_pl)
            #epoch_losses_rec.append(epoch_loss_rec)
            epoch_z = torch.cat(z_save,0).cpu().detach().numpy()

            plot_training_samples(transformed_images, reconstructed_images)
          
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            wandb.log({f'Train/Example_': [
                wandb.Image(Image.open(buf),  caption="Iteration_" + str(epoch))]})

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_pl_": epoch_loss_pl, '_step_': epoch})
            #wandb.log({"Train/Loss_Rec_": epoch_loss_rec, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           ,'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)
        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_pl': 0,
            task + '_loss_mse': 0,
            task + '_loss': 0,
        }
        test_total = 0

        attributes = []
        latent_codes = []
        with torch.no_grad():
            for x, labels, attr,_ in test_data:
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                x_rec, f_result = self.test_model(x)
                loss_rec = self.criterion_MSE(x_rec, x)
                loss_reg = 0#
                loss = self.criterion_rec(x_rec,x, f_result, labels, attr.to(self.device))
                loss_pl = self.criterion_PL(x_rec, x)

                #if self.num_classes == 1:
                #    acc = mean_accuracy(f_result['out_mlp'], labels)
                #else:
                #    acc = accuracy(f_result['out_mlp'], labels, average="macro", num_classes=self.num_classes)
                acc = 0

                metrics[task + '_loss_mse'] += loss_rec * x.size(0)
                metrics[task + '_loss'] += loss * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

                latent_codes.append(f_result['z'].cpu().numpy())
                attributes.append(attr)

        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)

        if epoch % 50 == 0:
            rl_metrics = compute_rl_metrics('',latent_codes,attributes,test_data.dataset.dataset.attributes_idx)
            metrics.update(rl_metrics) # add the rl_metrics

        # Metrics to wandb
        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)

            if 'loss' in metric_key:
                metric_score = metrics[metric_key] / test_total
                wandb.log({metric_name: metric_score, '_step_': epoch})
            else: # rl_metrics
                if metric_key == 'interpretability':
                    for attr_name in test_data.dataset.dataset.attributes_idx:
                        m_name = f'{metric_name}_{attr_name}'
                        metric_score = metrics[metric_key][attr_name][1]
                        wandb.log({m_name: metric_score, '_step_': epoch})
                    pass

                metric_score = metrics[metric_key]
                wandb.log({metric_name: metric_score, '_step_': epoch})

        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        #wandb.log({'Val/ Acc': acc[0].detach().cpu().numpy(), '_step_': epoch})

        if task == 'Val':
            plot_training_samples(x, x_rec)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            wandb.log({task + '/Example': [ wandb.Image(Image.open(buf), caption="Iteration_" + str(epoch))]})

        # Store best model
        epoch_val_loss = metrics[task + '_loss'] / test_total

        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = copy.deepcopy(model_weights)
                self.best_opt_weights = copy.deepcopy(opt_weights)

            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)
