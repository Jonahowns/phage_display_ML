from rbm_torch.models.pool_crbm_base import pool_CRBM

import torch.nn as nn
import torch
import math

from rbm_torch.utils.utils import Categorical
from rbm_torch.utils.utils import configure_optimizer
from torch_geometric.nn import GATv2Conv
import torch_geometric as tg
import torch.nn.functional as F

class variational_pcrbm(pool_CRBM):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision, meminfo=False)

        assert self.sampling_strategy in ["random", "stratified"]

        # standard deviation of inputs loss term
        self.ls = config["ls"]

    def neg_elbo(self, v, h, var_params):
        return self.energy(v, h).mean() + self.entropy(h, var_params)

    def sample_variational(self, size, variational_param_dict):
        samples = []
        for key in self.hidden_convolution_keys:
            z = torch.normal(0, 1, size=(size, self.convolution_topology[key]["number"],), device=self.device)
            samples.append(variational_param_dict[f"var_mean_{key}"] + z*torch.exp(variational_param_dict[f"var_log_sigma_{key}"]))
        return samples

    def fit_variational(self, v, norm_convergence=0.05, steps=5):
        var_params = torch.nn.ParameterDict({})
        for key in self.hidden_convolution_keys:
            var_params[f"var_mean_{key}"] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            var_params[f"var_log_sigma_{key}"] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))

        tmp_optim = torch.optim.SGD(var_params.values(), lr=0.1, momentum=0.5)
        grad_norm = 1.
        # while grad_norm > norm_convergence:  # test for convergence
        for _ in range(steps):
            htilde = self.sample_variational(v.shape[0], var_params)
            loss = self.neg_elbo(v, htilde, var_params)
            loss.backward()
            grad_norm = self.grad_norm(var_params)
            tmp_optim.step()
            tmp_optim.zero_grad()
        return var_params, htilde, loss.detach()

    def grad_norm(self, params):
        total_norm = torch.zeros((1,), device=self.device)
        for p in params.values():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2 / p.shape[0]
        return total_norm ** 0.5

    def entropy(self, h, var_params):
        entropy = torch.zeros((1,), device=self.device)
        denom = 0.
        for kid, key in enumerate(self.hidden_convolution_keys):
            entropy_key = -var_params[f"var_log_sigma_{key}"] - 0.5*math.log(2*torch.pi) - (h[kid]-var_params[f"var_mean_{key}"]).square()/(2*torch.square(torch.exp(var_params[f"var_log_sigma_{key}"])))
            entropy += entropy_key.sum((0, 1))
            denom += h[kid].shape[0] * h[kid].shape[1]

        return entropy/denom

    def forward(self, htilde):
        v = self.sample_from_inputs_v(self.compute_output_h(htilde))
        h = self.sample_from_inputs_h(self.compute_output_v(v))
        return v, h

    def training_step(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        vp, htilde, elbo_loss = self.fit_variational(one_hot)
        
        h_inputs = self.compute_output_v(one_hot)
        std_dev_h = torch.tensor(0., device=self.device)
        for h in h_inputs:
            std_dev_h += self.ls*torch.std(h.abs())

        vneg, hneg = self(htilde)

        reconstruction_error = 1 - (one_hot.argmax(-1) == vneg.argmax(-1)).double().mean(-1)
        seq_weights *= reconstruction_error

        free_energy = self.free_energy(one_hot)
        F_v = (self.free_energy(one_hot) * seq_weights / seq_weights.sum()).sum()  # free energy of training data
        F_vp = (self.free_energy(vneg) * seq_weights / seq_weights.sum()).sum()  # free energy of gibbs sampled visible states
        free_energy_diff = F_v - F_vp
        cd_loss = free_energy_diff

        loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss - std_dev_h


        logs = {"loss": loss,
                "neg_elbo": elbo_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/neg_elbo_loss", elbo_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs



class vr_pcrbm(variational_pcrbm):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision)

        # parameters from h to y layer
        for key in self.hidden_convolution_keys:
            self.register_parameter(f"M_{key}", nn.Parameter(torch.randn((self.convolution_topology[key]["number"],), device=self.device) * 0.05))

        # std deviation for our predictions
        self.register_parameter(f"sigma", nn.Parameter(torch.tensor([0.5], device=self.device)))

        # bias on y layer
        self.register_parameter("b", nn.Parameter(torch.zeros((1,), device=self.device)))

        h_flat_size = sum([self.convolution_topology[key]["number"] for key in self.hidden_convolution_keys])

        optimizer = config['optimizer']
        self.y_optimizer = configure_optimizer(optimizer)



        # self.predict_net = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(h_flat_size, h_flat_size//2),
        #     nn.Dropout(self.dr),
        #     nn.BatchNorm1d(h_flat_size//2),
        #     nn.ReLU(),
        #     nn.Linear(h_flat_size//2 , 1)
        # )

    @property
    def automatic_optimization(self):
        return False

    def neg_elbo_y(self, v, h, y, var_params):
        rbm_elbo = -(self.energy_h(h, sub_index=-1, remove_init=False) + self.bidirectional_weight_term(v, h, hidden_sub_index=-1)).mean() + self.entropy(h, var_params)

        expectation_suff_stats = torch.zeros((y.shape[0],), device=self.device)
        expectation_log_part = torch.zeros((y.shape[0],), device=self.device)
        for kid, key in enumerate(self.hidden_convolution_keys):
            bias = getattr(self, "b")
            y_sigma = getattr(self, "sigma")

            M = getattr(self, f"M_{key}")

            # Expectation of sufficient statistics
            var_sigma = torch.exp(var_params[f"var_log_sigma_{key}"])
            var_mu = var_params[f"var_mean_{key}"]
            du = h[kid] - var_mu

            y_expand = y.unsqueeze(1).expand(-1, var_mu.shape[0])

            t1 = -2. * torch.exp(-du.square()/(2*var_sigma.square())) * M * var_sigma.square()
            t2 = math.sqrt(math.pi/2) * var_sigma * (2*bias + 2*M*var_mu - y_expand) * torch.erf(du/(math.sqrt(2)*var_sigma))

            tmp = (y_expand * (t1+t2)) / (2 * math.sqrt(2*math.pi)*y_sigma.square()*var_sigma).unsqueeze(0)
            expectation_suff_stats += tmp.sum(1)

            # Expectation of log partition function at 2nd order approximation
            out = h[kid].matmul(M) + getattr(self, "b")

            expectation_log_part += out.square()/y_sigma.square() + torch.log(y_sigma) + 0.5*(M.square()/y_sigma.square() * h[kid] * (1 - h[kid])).sum(1)

        final = rbm_elbo - expectation_suff_stats.mean() + expectation_log_part.mean()
        return final

    def fit_variational_y(self, v, y, steps=5):
        var_params = torch.nn.ParameterDict({})
        for key in self.hidden_convolution_keys:
            var_params[f"var_mean_{key}"] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            var_params[f"var_log_sigma_{key}"] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))

        tmp_optim = torch.optim.SGD(var_params.values(), lr=0.1, momentum=0.5)
        # while grad_norm > norm_convergence:  # test for convergence
        for _ in range(steps):
            tmp_optim.zero_grad()
            htilde = self.sample_variational(v.shape[0], var_params)
            loss = self.neg_elbo_y(v, htilde, y, var_params)
            loss.backward()
            grad_norm = self.grad_norm(var_params)
            tmp_optim.step()
        return var_params, htilde, loss.detach()

    def regression_loss(self, y, htilde):
        suff_stats = torch.zeros(y.shape, device=self.device)
        log_part = torch.zeros(y.shape, device=self.device)

        sigma = getattr(self, "sigma")
        for kid, key in enumerate(self.hidden_convolution_keys):
            pred = htilde[kid].matmul(getattr(self, f"M_{key}")) + getattr(self, "b")

            suff_stats.add_(-y.square() / (2*sigma.square()) + y*pred/(sigma.square()))
            log_part.add_(pred.square() / (2 * sigma.square()) + torch.log(sigma))

        return log_part - suff_stats
        # return suff_stats - log_part

    def predict_y(self, htilde):
        mu = torch.zeros(htilde[0].shape[0], device=self.device)
        for kid, key in enumerate(self.hidden_convolution_keys):
            mu += htilde[kid].matmul(getattr(self, f"M_{key}")) + getattr(self, "b")

        sigma = getattr(self, "sigma")
        # return torch.normal(mu, sigma)
        return mu

    def training_step(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        vp, htilde, elbo_loss = self.fit_variational_y(one_hot, seq_weights.detach())

        # h_inputs = self.compute_output_v(one_hot)
        # std_dev_h = torch.tensor(0., device=self.device)
        # for h in h_inputs:
        #     # std_dev_h += self.ls * torch.std(h.abs())
        #     zero = torch.zeros_like(h, device=self.device)
        #     h_pos = torch.maximum(h, zero)
        #     h_neg = torch.minimum(h, zero).abs()
        #
        #     # h_abs = h.abs()
        #     z_score_h_pos = (h_pos - h_pos.mean(1, keepdim=True))/torch.std(h_pos, 1, keepdim=True)
        #     z_score_h_neg = (h_neg - h_neg.mean(1, keepdim=True))/torch.std(h_neg, 1, keepdim=True)
        #     # Euclidean distance b/t each hidden units normalized inputs
        #     dist_pos = torch.cdist(z_score_h_pos.T, z_score_h_pos.T).sum((0, 1))/(2 * h.shape[0] * h.shape[1])
        #     dist_neg = torch.cdist(z_score_h_neg.T, z_score_h_neg.T).sum((0, 1))/(2 * h.shape[0] * h.shape[1])
        #     dist_mix = torch.cdist(z_score_h_neg.T, z_score_h_pos.T).sum((0, 1))/(2 * h.shape[0] * h.shape[1])
        #
        #     std_dev_h += self.ls * (dist_pos + dist_neg + dist_mix)
        #
        # hflat = torch.concat(h_inputs, dim=1)
        #
        # net_preds = self.predict_net(hflat)


        ht = [x.detach().clone() for x in htilde]
        vneg, hneg = self(htilde)

        reconstruction_error = 1 - (one_hot.argmax(-1) == vneg.argmax(-1)).double().mean(-1)
        cd_weights = reconstruction_error   # * seq_weights

        F_v = (self.free_energy(one_hot) * cd_weights / cd_weights.sum()).sum()  # free energy of training data
        F_vp = (self.free_energy(vneg) * cd_weights / cd_weights.sum()).sum()  # free energy of gibbs sampled visible states
        free_energy_diff = F_v - F_vp
        # cd_loss = free_energy_diff/free_energy_diff.abs()*10

        # elbo = self.neg_elbo_y(one_hot, ht, seq_weights.detach(), vp)

        regression_loss = self.regression_loss(seq_weights.detach(), ht).mean()

        cd_loss = free_energy_diff + reg1 + reg2 + reg3 + bs_loss + gap_loss  # / free_energy_diff.abs() * 2 * regression_loss.detach().floor()

        ypred = self.predict_y(ht)

        mse = (ypred - seq_weights).square().mean()

        # net_mse = (net_preds - seq_weights).square().mean()

        # loss = net_mse + cd_loss + reg1 + reg2 + bs_loss

        # loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss + regression_loss * 100 #+ elbo - std_dev_h
        # loss = regression_loss + elbo + cd_loss + reg1 + reg2 + bs_loss

        self.opt_rbm.backward(cd_loss)
        self.y_rbm.backward(regression_loss)

        # self.manual_backward(cd_loss, self.opt_rbm)
        # self.manual_backward(regression_loss, self.opt_y)
        self.opt_rbm.step()
        self.opt_y.step()


        logs = {"cd_loss": cd_loss,
                "neg_elbo": elbo_loss.detach(),
                "train_free_energy": F_v.detach(),
                "regression_loss": regression_loss.detach(),
                # "mse": mse.detach(),
                "mse": mse.detach(),
                # "input_deviation_loss": std_dev_h.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/neg_elbo_loss", elbo_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ptl/MSE", mse.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/MSE", mse.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def validation_step(self, batch, batch_idx):
        return {"loss": torch.tensor([0.], device=self.device)}

    def predict(self, X):
        # Read in data
        reader =  Categorical(X, self.q, weights=None, max_length=self.v_num, molecule=self.molecule, device=self.device, one_hot=True)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        self.eval()
        with torch.no_grad():
            likelihood = []
            ys = []
            for i, batch in enumerate(data_loader):
                inds, seqs, one_hot, seq_weights = batch
                likelihood += self.likelihood(one_hot).detach().tolist()
                vp, htilde, elbo_loss = self.fit_variational_y(one_hot, seq_weights.detach(), steps=10)
                ys += self.predict_y(htilde)

        return X.sequence.tolist(), likelihood, ys

    def configure_optimizers(self):
        rbm_params = [getattr(self, "fields")]
        y_params = [getattr(self, "sigma"), getattr(self, "b")]
        for key in self.hidden_convolution_keys:
            rbm_params += [getattr(self, f"{key}_{x}") for x in ["W", "gamma+", "gamma-", "theta+", "theta-"]]
            y_params.append(getattr(self, f"M_{key}"))


        rbm_optim = self.optimizer(rbm_params, lr=self.lr, weight_decay=self.wd)
        y_optim = self.y_optimizer(y_params, lr=self.lr, weight_decay=self.wd)
        # optim = self.optimizer(self.weight_param)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lrf / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        rbm_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=rbm_optim, milestones=[decay_milestone], gamma=decay_gamma)
        y_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=y_optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=10)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        optim_dict = {"lr_scheduler": [rbm_scheduler, y_scheduler],
                      "optimizer": [rbm_optim, y_optim]}

        self.opt_rbm = rbm_optim
        self.opt_y = y_optim
        return self.opt_rbm, self.opt_y



class hybrid_pcrbm(pool_CRBM):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision, meminfo=False)

        h_flat_size = sum([self.convolution_topology[key]["number"] for key in self.hidden_convolution_keys])

        self.latent_dim = config["latent_dim"]
        # self.node_depth = config["node_depth"]
        self.adj_depth = config["adj_depth"]
        self.predictor_depth = config["predictor_depth"]

        self.feature_num = self.v_num - 7 + 1
        self.node_size = h_flat_size

        # self.automatic_optimization = False

        # self.node_conv_net = nn.Sequential(
        #     # nn.BatchNorm1d(self.node_size),
        #     nn.ReLU())

        # self.adjacency_net = nn.Sequential(*self.make_adj_net(depth=self.adj_depth))
        #
        # self.predictor_net = nn.Sequential(*self.make_predictor_net(self.latent_dim, depth=self.predictor_depth))
        # self.regression_loss = nn.MSELoss()

        self.cnn = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(1, 512, kernel_size=(20,), stride=(1,), padding=0),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=(20,), stride=(1,), padding=0),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=(10,), stride=(1,), padding=0),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=(10,), stride=(1,), padding=0),
            nn.GELU(),
        )
        self.cnn_lin = torch.nn.Sequential(
            nn.BatchNorm1d(64*44),
            nn.Linear(64*44, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )

        # GCN Components
        # self.c1 = GATv2Conv(-1, self.latent_dim)
        # self.c1 = GATv2Conv(-1, 3 * self.latent_dim)
        # self.c2 = GATv2Conv(3 * self.latent_dim, 2 * self.latent_dim)
        # self.c3 = GATv2Conv(2 * self.latent_dim, self.latent_dim)
        # self.c4 = GATv2Conv(self.latent_dim, self.latent_dim)
        # self.c5 = GATv2Conv(self.latent_dim, self.latent_dim)
        # self.c6 = GATv2Conv(self.latent_dim, self.latent_dim)

        # self.bn_c1 = torch.nn.BatchNorm1d(self.latent_dim)
        # self.bn_c1 = torch.nn.BatchNorm1d(3 * self.latent_dim)
        # self.bn_c2 = torch.nn.BatchNorm1d(2 * self.latent_dim)
        # self.bn_c3 = torch.nn.BatchNorm1d(self.latent_dim)
        # self.bn_c4 = torch.nn.BatchNorm1d(self.latent_dim)
        # self.bn_c5 = torch.nn.BatchNorm1d(self.latent_dim)
        # self.bn_c6 = torch.nn.BatchNorm1d(self.latent_dim)

        # self.aggregator = tg.nn.PowerMeanAggregation(1.0, learn=False)
        # self.aggregator = tg.nn.MeanAggregation()

        # self.predict_net = nn.Sequential(
        #     nn.Linear(self.node_size, self.node_size // 2),
        #     nn.Dropout(self.dr),
        #     nn.BatchNorm1d(self.node_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.node_size // 2, 1)
        # )

        # self.combined_embedding_net = nn.Sequential(
        #     nn.Linear(int(self.node_size*(self.node_size-1)/2), self.node_size),
        #     # nn.BatchNorm1d(self.node_size),
        #     nn.ReLU()
        # )

        self.save_hyperparameters()

    def gcn_forward(self, X):
        x = self.compute_output_v(X)
        x = torch.concat(x, dim=1)
        zero = torch.zeros_like(x, device=self.device)
        abs_x = torch.maximum(x, zero) + torch.minimum(x, zero).abs()
        nodes = self.node_conv_net(abs_x)
        flat_nodes = nodes.view(nodes.shape[0] * self.node_size, -1)

        adj_probs = self.adjacency_net(nodes)
        adj_encoding = torch.bernoulli(adj_probs)

        edges = self.make_adjacency_matrix(adj_encoding)

        edge_embedding = self.combined_embedding_net(adj_encoding).reshape(nodes.shape[0] * self.node_size, -1)
        combined_embedding = edge_embedding * flat_nodes

        x = self.c1(combined_embedding, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        # x = self.bn_c1(x)

        x = self.aggregator(x, torch.arange(0, nodes.shape[0], 1 / (self.node_size), device=self.device).long())

        # Finally, Our predictor network
        return self.predictor_net(x).squeeze(1)

    def mse_loss(self, preds, targets):
        return (preds - targets).square().mean()

    def robust_loss(self, preds, targets, alpha, c):
        # from: https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf
        # Author: Jonathan T Barton
        x = preds - targets
        ab_alpha = abs(alpha - 2)
        return (ab_alpha / alpha * (torch.pow((x / c).square() / ab_alpha + 1, alpha / 2) - 1)).mean()

    def training_step(self, batch, batch_idx):
        if self.sample_stds is not None:
            inds, seqs, one_hot, seq_weights, stds = batch
        else:
            inds, seqs, one_hot, seq_weights = batch

        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        vneg, hneg, vpos, hpos = self(one_hot)

        # reconstruction_error = 1 - (one_hot.argmax(-1) == vneg.argmax(-1)).double().mean(-1)
        # cd_weights = reconstruction_error  # * seq_weights

        F_v = (self.free_energy(one_hot)).sum()  # free energy of training data
        F_vp = (self.free_energy(vneg)).sum()  # free energy of gibbs sampled visible states
        free_energy_diff = F_v - F_vp
        cd_loss = free_energy_diff  #/ free_energy_diff.abs()

        # gcn_preds = self.gcn_forward(one_hot)
        x = self.compute_output_v(one_hot)
        x = torch.concat(x, dim=1)
        zero = torch.zeros_like(x, device=self.device)
        abs_x = torch.maximum(x, zero) + torch.minimum(x, zero).abs()
        conv_out = self.cnn(abs_x.unsqueeze(1))

        gcn_preds = self.cnn_lin(conv_out.flatten(1))

        adj_start_epoch = 100
        y_steps = 150
        if self.sample_stds is not None and self.current_epoch >= adj_start_epoch:
            if self.current_epoch == adj_start_epoch:
                if batch_idx == 0:
                    self.ystar = torch.zeros((self.training_data.index.__len__(),), device=self.device)
                current_y = seq_weights
            else:
                current_y = self.ystar[inds]

            ys = self.adjust_labels(gcn_preds, current_y, seq_weights, stds, steps=y_steps, stds_allowed=2.0)
            if True in torch.isnan(ys):
                print("Label Adjustment Produced Nan")
                exit(1)

            self.ystar[inds] = ys
            diff_y_l2 = torch.pow(ys - seq_weights, 2).mean()
        else:
            ys = seq_weights
            diff_y_l2 = torch.zeros((1,), device=self.device)

        gcn_mse = self.mse_loss(gcn_preds, ys)
        gcn_robust = self.robust_loss(gcn_preds, ys, -1, 2 * stds)
        gcn_mae = (gcn_preds - ys).abs().mean() #+ self.current_epoch/self.epochs

        loss = gcn_mae + 0.1*cd_loss + reg1 + reg2 + reg3 + bs_loss

        # opt = self.optimizers()
        # opt.step(None, loss_array=[gcn_mae, cd_loss], ranks=[1, 1], feature_map=None)
        # opt.zero_grad()

        # loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss - std_dev_h + regression_loss*1000 + elbo
        # loss = regression_loss + elbo + cd_loss + reg1 + reg2 + bs_loss

        logs = {"loss": loss,
                "train_free_energy": F_v.detach(),
                "train_gcn_mse": gcn_mse.detach(),
                "train_gcn_mae": gcn_mae.detach(),
                "train_gcn_robust": gcn_robust.detach(),
                "diff_y_l2": diff_y_l2.detach(),
                "cd_loss": cd_loss.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_MSE",  logs["train_gcn_mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_MAE",  logs["train_gcn_mae"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_robust",  logs["train_gcn_robust"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def validation_step(self, batch, batch_idx):
        if self.sample_stds is not None:
            inds, seqs, one_hot, seq_weights, stds = batch

        else:
            inds, seqs, one_hot, seq_weights = batch

        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        vneg, hneg, vpos, hpos = self(one_hot)

        reconstruction_error = 1 - (one_hot.argmax(-1) == vneg.argmax(-1)).double().mean(-1)
        cd_weights = reconstruction_error  # * seq_weights

        F_v = (self.free_energy(one_hot) * cd_weights / cd_weights.sum()).sum()  # free energy of training data
        F_vp = (self.free_energy(vneg) * cd_weights / cd_weights.sum()).sum()  # free energy of gibbs sampled visible states
        free_energy_diff = F_v - F_vp
        cd_loss = free_energy_diff / free_energy_diff.abs()

        x = self.compute_output_v(one_hot)
        x = torch.concat(x, dim=1)
        zero = torch.zeros_like(x, device=self.device)
        abs_x = torch.maximum(x, zero) + torch.minimum(x, zero).abs()
        conv_out = self.cnn(abs_x.unsqueeze(1))

        gcn_preds = self.cnn_lin(conv_out.flatten(1))

        gcn_mse = self.mse_loss(gcn_preds, seq_weights)
        gcn_mae = (gcn_preds - seq_weights).abs().mean()
        gcn_robust = self.robust_loss(gcn_preds, seq_weights, -1, 2*stds)

        loss = gcn_mae + cd_loss + reg1 + reg2 + reg3 + bs_loss

        logs = {"loss": [gcn_mae, cd_loss],
                "val_free_energy": F_v.detach(),
                "val_gcn_mse": gcn_mse.detach(),
                "val_gcn_mae": gcn_mae.detach(),
                "val_gcn_robust": gcn_robust.detach(),
                **reg_dict
                }

        self.log("ptl/val_free_energy", logs["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_MSE", logs["val_gcn_mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_MAE", logs["val_gcn_mae"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_robust", logs["val_gcn_robust"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def make_adj_net(self, depth=3):
        fcn_start_size = self.node_size
        fcn_end_size = int((self.node_size*(self.node_size - 1)) / 2)

        fcn_size = [fcn_start_size]
        # if self.network_layers > 1:
        for i in range(depth - 1):
            in_size = fcn_size[-1]
            fcn_size.append(in_size // 2)

        fcn_size.append(fcn_end_size)

        network = []
        for i in range(depth - 1):
            network.append(nn.Dropout(self.dr))
            network.append(nn.Linear(fcn_size[i], fcn_size[i + 1], dtype=torch.get_default_dtype()))
            # network.append(nn.BatchNorm1d(fcn_size[i + 1]))
            network.append(nn.LeakyReLU())

        network.append(nn.Dropout(self.dr))
        network.append(nn.Linear(fcn_size[-2], fcn_size[-1], dtype=torch.get_default_dtype()))
        network.append(nn.Sigmoid())  # nice values b/t 0 and 1
        return network

    def make_predictor_net(self, in_features, dropout=0.1, depth=1):
        net = []
        for i in range(depth - 1):  # build many linear layers halving the number of units each time
            out_features = in_features // 2
            net.append(nn.Linear(in_features, out_features))
            in_features = out_features

            net.append(nn.LeakyReLU())  # activation function
            net.append(nn.Dropout(p=self.dr))  # dropout

        # Last layer, since its regression, 1 output feature
        net.append(nn.Linear(in_features, 1))
        return net

    def make_adjacency_matrix(self, adj_encoding):
        """Converts probabilities of edges and converts to a full edge list"""
        # Returns Tuple of edge tensors


        # location of all edges as sampled above
        global_edges = torch.argwhere(adj_encoding)

        # decode flattened upper triangular matrix into linear i and j indices
        # from: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        global_edges_batch = global_edges[:, 0].clone()
        global_edges[:, 0] = global_edges[:, 1]
        global_edges[:, 0] = self.node_size - 2 - (torch.sqrt(-8 * global_edges[:, 0] + 4 * self.node_size * (self.node_size - 1) - 7) / 2.0 - 0.5).long()
        global_edges[:, 1] = global_edges[:, 1] + global_edges[:, 0] + 1 - self.node_size * (self.node_size - 1) / 2 + (self.node_size - global_edges[:, 0]) * ((self.node_size - global_edges[:, 0]) - 1) / 2

        global_edges.add_(global_edges_batch.unsqueeze(1) * self.node_size)

        # for splitting into subtensors
        # vals, counts = torch.unique(global_edges[:, 0], return_counts=True)
        # edges = torch.tensor_split(global_edges.T, torch.cumsum(counts, 0), dim=1)

        return global_edges.T.long()

    def adjust_labels(self, preds, current_labels, original_labels, stds, steps=2, lr=0.1, stds_allowed=1.):
        ystar = torch.nn.Parameter(current_labels.detach().clone())
        p = preds.detach().clone()
        ol = original_labels.detach().clone()
        cl = current_labels.detach().clone()

        adj_stds = stds*stds_allowed

        max_dev = ol + adj_stds
        min_dev = ol - adj_stds

        optimizer = torch.optim.SGD(lr=lr, params=[ystar])
        loss = p - ystar
        for _ in range(steps):
            likeli = 1 / adj_stds * math.sqrt(2 * math.pi) * torch.exp(-torch.pow((ystar - cl) / (2 * adj_stds), 2)) * (1 + torch.erf(loss * (ystar - cl) / (adj_stds * math.sqrt(2))))
            log_likeli = torch.log(likeli) * adj_stds
            nll_loss = -log_likeli.mean()
            optimizer.zero_grad()
            nll_loss.backward()
            optimizer.step()

        return torch.clip(ystar, min_dev, max_dev).detach().type(torch.get_default_dtype())
