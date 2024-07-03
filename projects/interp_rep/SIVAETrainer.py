import pandas as pd
import numpy as np
from core.Trainer import Trainer
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR

from time import time
import wandb
from dl_utils.config_utils import *
import logging
from model_zoo.soft_intro_vae_daniel import *
import matplotlib.pyplot as plt

from optim.losses.image_losses import compute_reg_loss
from optim.metrics.rl_metrics import *
import io
from PIL import Image
from dl_utils.vizu_utils import plot_training_samples
import yaml
import torch
from torchmetrics.functional import confusion_matrix, accuracy
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

class PTrainer(Trainer):
    """
    Train Soft-Intro VAE for image datasets
    Code based on: https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae/

    T. Daniel and A. Tamar. Soft-introvae: Analyzing and improving the introspective variational autoencoder.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4391–4400, 2021.
    """
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.optimizer_e = Adam(model.encoder.parameters(), lr=training_params['optimizer_params']['lr'])
        self.optimizer_d = Adam(model.decoder.parameters(), lr=training_params['optimizer_params']['lr'])

        self.e_scheduler = MultiStepLR(self.optimizer_e, milestones=(100,), gamma=0.1)
        self.d_scheduler = MultiStepLR(self.optimizer_d, milestones=(100,), gamma=0.1)

        self.scale = 1 / (training_params['input_size'][1] ** 2)  # normalize by images size (channels * height * width)
        self.gamma_r = 1e-8
        self.beta_kl = training_params['beta_kl'] if 'beta_kl' in training_params.keys() else 1.0
        self.beta_rec = training_params['beta_rec'] if 'beta_rec' in training_params.keys() else 0.5
        self.beta_neg = training_params['beta_neg'] if 'beta_neg' in training_params.keys() else 128.0
        #self.beta_neg = 128.0

        self.reg_loss = training_params['reg_loss'] if 'reg_loss' in training_params.keys() else 0
        self.factor = training_params['factor'] if 'factor' in training_params.keys() else 10.0
        self.loss_type = training_params['loss_type'] if 'loss_type' in training_params.keys() else 'mse'
        self.annealing = training_params['annealing'] if 'annealing' in training_params.keys() else 1
        self.annealing_mse = training_params['annealing_mse'] if 'annealing_mse' in training_params.keys() else 1

        mlp_params = training_params['mlp'] if 'mlp' in training_params.keys() else None
        if mlp_params is not None:

            stream_file = open(mlp_params['mlp_config'], 'r')
            self.mlp_config = yaml.load(stream_file, Loader=yaml.FullLoader)

            model_class = import_module(self.mlp_config['module_name'], self.mlp_config['class_name'])
            mlp_model = model_class(**(self.mlp_config['params']))
            self.mlp_model = mlp_model.to(self.device)
            self.dict_classes = self.mlp_config['params']['dict_classes']
            self.num_classes = len(self.dict_classes)  # if len(self.dict_classes) > 2 else 1
            self.weights = torch.tensor([0.5, 0.5]) if self.num_classes == 2 else torch.ones([1, self.num_classes])
            self.phi = self.mlp_config['phi']
            self.epoch = self.mlp_config['epoch']
        else:
            self.mlp_model = None



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

        epoch_losses = []
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()

            diff_kls, batch_kls_real, batch_kls_fake, batch_kls_rec, batch_rec_errs, batch_exp_elbo_f,\
            batch_exp_elbo_r, count_images = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

            batch_loss_reg, batch_loss_mlp = 0.0, 0.0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images

                b, c, w, h = images.shape

                count_images += b
                noise_batch = torch.randn(size=(b, self.model.zdim)).to(self.device)
                #noise_batch = torch.randn(size=(b, 128)).to(self.device)
                real_batch = transformed_images.to(self.device)

                # =========== Update E ================
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

                fake = self.model.sample(noise_batch)

                real_mu, real_logvar = self.model.encode(real_batch)
                z = reparameterize(real_mu, real_logvar)
                rec = self.model.decoder(z)

                #annealing = epoch / self.training_params['nr_epochs'] if epoch > 0 else 0 # annealing applied only if loss_type = pl

                #loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type= 'mse', reduction="mean")
                if self.loss_type == 'pl':
                    loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type='mse', reduction="mean")
                    pl_error = self.criterion_PL(real_batch, rec)
                    loss_rec = self.annealing_mse * loss_rec + self.annealing * pl_error
                else:
                    loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=self.loss_type, reduction="mean")

                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                rec_rec, z_dict = self.model(rec.detach(), deterministic=False)
                rec_mu, rec_logvar, z_rec = z_dict['z_mu'], z_dict['z_logvar'], z_dict['z']
                rec_fake, z_dict_fake = self.model(fake.detach(), deterministic=False)
                fake_mu, fake_logvar, z_fake = z_dict_fake['z_mu'], z_dict_fake['z_logvar'], z_dict_fake['z']

                kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
                kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")


                loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type= self.loss_type, reduction="none")
                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)

                # PL loss
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(rec, rec_rec)
                    loss_rec_rec_e = self.annealing_mse * loss_rec_rec_e + self.annealing * pl_error

                loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type= self.loss_type, reduction="none")
                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)
                # PL loss
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(fake, rec_fake)
                    loss_rec_fake_e = self.annealing_mse * loss_rec_fake_e + self.annealing * pl_error

                expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

                lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)
                loss_reg = self.reg_loss * compute_reg_loss(z, data[2], self.factor)

                lossE = lossE_real + lossE_fake + loss_reg

                self.optimizer_e.zero_grad()
                lossE.backward() # propagate all of the losses in the encoder
                self.optimizer_e.step()

                # ========= Update D ==================

                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = True

                fake = self.model.sample(noise_batch)
                rec = self.model.decoder(z.detach())

                loss_rec = calc_reconstruction_loss(real_batch, rec,  loss_type= self.loss_type, reduction="mean")
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(real_batch, rec)
                    loss_rec = self.annealing_mse * loss_rec + self.annealing * pl_error

                rec_mu, rec_logvar = self.model.encode(rec)
                z_rec = reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar = self.model.encode(fake)
                z_fake = reparameterize(fake_mu, fake_logvar)

                rec_rec = self.model.decode(z_rec.detach())
                rec_fake = self.model.decode(z_fake.detach())

                loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec,  loss_type= self.loss_type, reduction="mean")
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(rec, rec_rec)
                    loss_rec_rec = self.annealing_mse * loss_rec_rec + self.annealing * pl_error

                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake,  loss_type= self.loss_type, reduction="mean")
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(fake, rec_fake)
                    loss_fake_rec = self.annealing_mse * loss_fake_rec + self.annealing * pl_error

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

                lossD = self.scale * (loss_rec * self.beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                self.optimizer_d.zero_grad()
                lossD.backward()
                self.optimizer_d.step()
                if torch.isnan(lossD) or torch.isnan(lossE):
                    print('is non for D')
                    raise SystemError
                if torch.isnan(lossE):
                    print('is non for E')
                    raise SystemError


                diff_kls += -lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item() * images.shape[0]
                batch_kls_real += lossE_real_kl.data.cpu().item() * images.shape[0]
                batch_kls_fake += lossD_fake_kl.cpu().item() * images.shape[0]
                batch_kls_rec += lossD_rec_kl.data.cpu().item() * images.shape[0]
                batch_rec_errs += loss_rec.data.cpu().item() * images.shape[0]

                batch_exp_elbo_f += expelbo_fake.data.cpu() * images.shape[0]
                batch_exp_elbo_r += expelbo_rec.data.cpu() * images.shape[0]
                batch_loss_reg += loss_reg * images.shape[0]

                #if self.mlp_model is not None:
                #    batch_loss_mlp += loss_mlp * images.shape[0]

            epoch_loss_d_kls = diff_kls / count_images if count_images > 0 else diff_kls
            epoch_loss_kls_real = batch_kls_real / count_images if count_images > 0 else batch_kls_real
            epoch_loss_kls_fake = batch_kls_fake / count_images if count_images > 0 else batch_kls_fake
            epoch_loss_kls_rec = batch_kls_rec / count_images if count_images > 0 else batch_kls_rec
            epoch_loss_rec_errs = batch_rec_errs / count_images if count_images > 0 else batch_rec_errs
            epoch_loss_exp_f = batch_exp_elbo_f / count_images if count_images > 0 else batch_exp_elbo_f
            epoch_loss_exp_r = batch_exp_elbo_r / count_images if count_images > 0 else batch_exp_elbo_r
            epoch_loss_reg = batch_loss_reg / count_images if count_images > 0 else batch_loss_reg
            #epoch_loss_mlp = batch_loss_mlp / count_images if count_images > 0 else batch_loss_mlp

            epoch_losses.append(epoch_loss_rec_errs)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss_rec_errs, end_time - start_time, count_images))
            wandb.log({"Train/Loss_DKLS": epoch_loss_d_kls, '_step_': epoch})
            wandb.log({"Train/Loss_REAL": epoch_loss_kls_real, '_step_': epoch})
            wandb.log({"Train/Loss_FAKE": epoch_loss_kls_fake, '_step_': epoch})
            wandb.log({"Train/Loss_REC": epoch_loss_kls_rec, '_step_': epoch})
            wandb.log({"Train/Loss_REC_ERRS": epoch_loss_rec_errs, '_step_': epoch})
            wandb.log({"Train/Loss_EXP_F": epoch_loss_exp_f, '_step_': epoch})
            wandb.log({"Train/Loss_EXP_R": epoch_loss_exp_r, '_step_': epoch})
            wandb.log({"Train/Loss_REG": epoch_loss_reg, '_step_': epoch})
            #wandb.log({"Train/Loss_MLP": epoch_loss_mlp, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')

            #if self.mlp_model is not None:
            #    torch.save(
            #        {'model_weights': self.mlp_model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
            #            , 'epoch': epoch}, self.client_path + '/latest_model_head.pt')

            plot_training_samples(transformed_images, rec)

            buf = io.BytesIO()
            plt.savefig(buf, format = 'png')
            buf.seek(0)
            wandb.log({'Train/Example_': [wandb.Image(Image.open(buf), caption="Iteration_" + str(epoch))]})
            #wandb.log({'Train/Example_': [wandb.Image(diffp, caption="Iteration_" + str(epoch))]})


            self.test(self.model.state_dict(), self.val_ds, 'Val', [self.optimizer_e.state_dict(),
                                                                    self.optimizer_d.state_dict()], epoch)

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

        if self.mlp_model is not None:
            self.mlp_model.eval()

        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
            task + '_loss_mlp': 0
        }
        test_total = 0

        annealing = epoch / self.training_params['nr_epochs'] if epoch > 0 else 0  # annealing applied only if loss_type = pl
        attributes = []
        latent_codes = []
        labels, predictions = [], [] # used if self.mlp_model is not None

        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_, z_rec = self.test_model(x)
                loss_rec = calc_reconstruction_loss(x_, x, loss_type=self.loss_type)
                if self.loss_type == 'pl':
                    pl_error = self.criterion_PL(x_, x)
                    loss_rec = self.annealing_mse * loss_rec + self.annealing * pl_error

                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

                if self.mlp_model is not None:
                    y_hat = self.mlp_model(z_rec['z'])
                    y = data[1].to(self.device)
                    if self.num_classes == 2:
                        y = (data[1].to(self.device) >= 1) * 1
                    loss_mlp = F.cross_entropy(y_hat, y, reduction="mean", weight=self.weights.to(self.device))
                    metrics[task + '_loss_mlp'] += loss_mlp.item() * x.size(0)
                    labels.append(y.cpu().numpy())
                    predictions.append(y_hat.cpu().numpy())

                latent_codes.append(z_rec['z'].cpu().numpy())
                attributes.append(data[2])

        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)

        if self.mlp_model is not None:
            labels = np.concatenate(labels, 0)
            predictions = np.concatenate(predictions, 0)

        if task == 'Val':

            plot_training_samples(x, x_)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            wandb.log({task + '/Example_': [wandb.Image(Image.open(buf), caption="Iteration_" + str(epoch))]})
            #wandb.log({task + '/Example_': [wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

            if epoch % 50 == 0:
                rl_metrics = compute_rl_metrics('', latent_codes, attributes, test_data.dataset.dataset.attributes_idx)
                metrics.update(rl_metrics)  # add the rl_metrics
                if self.mlp_model is not None:
                    metrics.update({'AUROC': roc_auc_score(labels, np.argmax(predictions,axis=1))})

            for metric_key in metrics.keys():
                metric_name = task + '/' + str(metric_key)

                if 'loss' in metric_key:
                    metric_score = metrics[metric_key] / test_total
                    wandb.log({metric_name: metric_score, '_step_': epoch})
                else:  # rl_metrics
                    if metric_key == 'interpretability':
                        for attr_name in test_data.dataset.dataset.attributes_idx:
                            m_name = f'{metric_name}_{attr_name}'
                            metric_score = metrics[metric_key][attr_name][1]
                            wandb.log({m_name: metric_score, '_step_': epoch})
                    else:
                        wandb.log({metric_name: metrics[metric_key], '_step_': epoch})

            wandb.log({'lr': self.optimizer_e.param_groups[0]['lr'], '_step_': epoch})
            epoch_val_loss = metrics[task + '_loss_rec'] / test_total

        #if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
                torch.save({'model_weights': model_weights, 'optimizer_e_weights': opt_weights[0],
                            'optimizer_d_weights': opt_weights[1], 'epoch': epoch},
                           self.client_path + '/best_model.pt')
                if self.mlp_model is not None:
                    torch.save({'model_weights': self.mlp_model.state_dict(), 'optimizer_weights': self.optimizer_e.state_dict()
                               , 'epoch': epoch}, self.client_path + '/best_model_head.pt')
            self.early_stop = self.early_stopping(epoch_val_loss)
            self.e_scheduler.step(epoch_val_loss)
            self.d_scheduler.step(epoch_val_loss)
