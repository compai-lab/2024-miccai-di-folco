import logging
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import torch
import json
from torch.nn import L1Loss, MSELoss
#
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import PeakSignalNoiseRatio
from dl_utils.config_utils import *
#
import lpips
from core.DownstreamEvaluator import DownstreamEvaluator
from optim.metrics.rl_metrics import *
from dl_utils.vizu_utils import *
import io
from PIL import Image
from pathlib import Path
import yaml

class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, mlp_config=None):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.attributes_dict = test_data_dict.dataset.dataset.attributes_dict
        self.attributes_idx = test_data_dict.dataset.dataset.attributes_idx
        self.name = name

        self.criterion_MSE = MSELoss().to(self.device)
        self.save_MSE = MSELoss(reduce=False).to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=False, lpips=True).to(self.device)

        if mlp_config is not None:
            stream_file = open(mlp_config, 'r')
            self.mlp_config = yaml.load(stream_file, Loader=yaml.FullLoader)
            model_class = import_module(self.mlp_config['module_name'], self.mlp_config['class_name'])
            mlp_model = model_class(**(self.mlp_config['params']))
            self.mlp_model = mlp_model.to(self.device)
            self.dict_classes = self.mlp_config['params']['dict_classes']
            self.num_classes = len(self.dict_classes)  # if len(self.dict_classes) > 2 else 1
            self.results_folder = self.checkpoint_path + '/'

            if Path(self.checkpoint_path + '/best_model_head.pt').exists():
                mlp_weight = torch.load(self.checkpoint_path + '/best_model_head.pt', map_location = torch.device(self.device))
                self.mlp_model.load_state_dict(mlp_weight['model_weights'])
        else:
            self.mlp_model = None

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        self.model.load_state_dict(global_model)
        self.model.eval()

        latent_codes, full_attributes, predictions, labels, rec_error = self.compute_latent_representations()
        rl_metrics = compute_rl_metrics(self.checkpoint_path, latent_codes.detach().cpu().numpy(), full_attributes, self.attributes_idx)

        #  Interpretability metrics
        df = pd.DataFrame(rl_metrics['interpretability'])
        tbl = wandb.Table(data=df)
        wandb.log({f"{self.name}/Interpretability metrics": tbl})

        df_metrics = pd.DataFrame()
        for key in rl_metrics.keys():
            if key != 'interpretability':
                df_metrics[key] = [rl_metrics[key]]


        for k in rec_error.keys():
            df_metrics[k] = np.mean(rec_error[k])

        tbl = wandb.Table(data=df_metrics)
        wandb.log({f"{self.name}/Metrics": tbl})

        dim_list = [rl_metrics['interpretability'][K][0] for K in rl_metrics['interpretability'].keys() if 'mean' not in K]

        rec_error.pop('MSE')
        save_df = pd.DataFrame(rec_error, columns = list(rec_error.keys()))
        
        print(self.checkpoint_path + f'/{self.name}_error.csv')
        save_df.to_csv(self.checkpoint_path + f'/{self.name}_error.csv')

        with open(self.checkpoint_path + f'/results_dict.json', 'w') as fp:
            json.dump(rl_metrics, fp)


        self.show_latent_space(latent_codes, dim_list = dim_list)

        if self.mlp_model is not None:
            self.prediction_task()

    def show_latent_space(self, latent_codes, dim_list = [0,1,2], dim_plot_2d = [0,1]):

        fig = plot_latent_reconstructions(self.model, self.test_data_dict, self.device, num_points=8)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        wandb.log({f'{self.name}/Reconstruction examples': [
            wandb.Image(Image.open(buf), caption=f'Test_reconstruction')]})

        range_value = 13.0
        fig = plot_latent_interpolations(self.model,latent_codes[:1,:], dim_list=dim_list,
                                         num_points=4, range_value=range_value)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        

        wandb.log({f'{self.name}/_Latent dimensions': [
             wandb.Image(Image.open(buf), caption= f'{range_value}' + 'Latent_dim_' + '_'.join(str(dim) for dim in dim_list))]})
        
 
    def compute_latent_representations(self):
        logging.info("################ Show latent space #################")

        # self.model.load_state_dict(global_model)
        # self.model.eval()

        dataset = self.test_data_dict

        idx = 0
        nc = self.model.nc

        latent_codes = []
        labels, predictions = [], []
        attributes, full_attributes = [], []


        ssim_ = {i: [] for i in range(nc)}
        psnr_= {i: [] for i in range(nc)}
        lpips_ = {i: [] for i in range(nc)}
        mse_loss = []


        PSNR = PeakSignalNoiseRatio().to(self.device)

        with torch.no_grad():
            for data, label, attr, full_attr in dataset:

                nr_slices, c, width, height, = data.shape
                x = data.view(nr_slices, c, width, height)

                x = x.to(self.device)
                rec, f_result = self.model(x)

                MSE = self.criterion_MSE(rec,x)
                save_MSE = self.save_MSE(rec,x)

                tmp = save_MSE.detach().cpu().numpy()
                save_MSE = [np.mean(tmp[i,:,:,:]) for i in range(tmp.shape[0])]
                mse_loss.append(save_MSE)

                for i in range(np.shape(data)[0]):
                    for N in range(nc):
                        x_i = x[i, N:N+1].cpu().detach().numpy()
                        x_rec_i = rec[i,N:N+1].cpu().detach().numpy()
                        S = ssim(x_rec_i, x_i, channel_axis=0 ,data_range=1.0)
                        ssim_[N].append(S)

                        psnr_[N].append(PSNR(rec[i,N], x[i,N]).cpu().detach().numpy().item())

                        lpips_value = torch.squeeze(self.l_pips_sq(torch.squeeze(rec[i,N]), torch.squeeze(x[i,N]), normalize=True, retPerLayer=False))
                        lpips_[N].append(lpips_value.cpu().detach().numpy().item())

                if len(f_result['z'].size()) > 2:
                    latent_codes.append(torch.squeeze(f_result['z']))
                else:
                    latent_codes.append(f_result['z'])
                attributes.append(attr.detach().cpu().numpy())
                full_attributes.append(full_attr.detach().cpu().numpy())
                labels.append(label)

        latent_codes = torch.cat(latent_codes,0)
        attributes = np.concatenate(attributes, 0)
        full_attributes = np.concatenate(full_attributes,0)
        labels = torch.cat(labels,0)
        if nc == 1:
            rec_error = {'MSE': np.concatenate(mse_loss,0), 'SSIM': ssim_[0], 'PSNR': psnr_[0], 'LPIPS': lpips_[0]}
        else:
            rec_error = {'MSE': np.concatenate(mse_loss,0),
                         'SSIM_ED': ssim_[0], 'SSIM_ES': ssim_[1],
                         'PSNR_ED': psnr_[0], 'PSNR_ES': psnr_[1],
                         'LPIPS_ED': lpips_[0], 'LPIPS_ES': lpips_[1]}


        #rec_error = np.mean(mse_loss)
        z_ = latent_codes.detach().cpu().numpy()

        return latent_codes, full_attributes, predictions, labels, rec_error

    def prediction_task(self):

        self.mlp_model.eval()

        pred = []
        labels = []

        with torch.no_grad():
            for data, label, attr, full_attr in self.test_data_dict:
                x = data.to(self.device)
                y = label.to(self.device)

                y_hat = self.mlp_model(x)

                pred.append(y_hat.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())

        pred = np.concatenate(pred, 0)
        labels = np.concatenate(labels, 0)

        if self.num_classes == 2:
            labels = (labels >= 1).astype(int)

        fig, df_acc = plot_conf_mat(pred, labels,
                                    self.dict_classes, False)

        tbl = wandb.Table(data=df_acc)
        wandb.log({"Test/Metrics": tbl})

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        wandb.log({'Test/Confusion matrix': [wandb.Image(Image.open(buf))]})

        index = 1
        labels_name = list(self.dict_classes.keys())
        attribution = AttributionLatentY(self.test_data_dict, labels_name, self.model,
                                         self.mlp_model, index, self.results_folder, self.device)

        fig_global, fig_local = attribution.visualization()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        wandb.log({'Test' + f'/Attribution_global': [
            wandb.Image(Image.open(buf), caption=f'Test_reconstruction')]})

        attribution_score, _, _ = attribution.attribution()

        df_scores = pd.DataFrame()
        for i in range(self.num_classes):
            scores = attribution_score[i]
            columns_names = [f + f'_{i}' for f in attribution.feature_names]
            df = pd.DataFrame(scores[:, :len(columns_names)], columns=columns_names)

            df_scores = pd.concat([df_scores, df], axis=1)
        print(self.checkpoint_path)

        run = self.checkpoint_path.split('/')[-1]

        if not os.path.exists(self.results_folder + f'/shap/'):
            os.makedirs(self.results_folder + f'/shap/')
        output_path = self.results_folder + f'/shap/values_{run}.csv'
        print(output_path)
        df_scores.to_csv(output_path, index=False)
