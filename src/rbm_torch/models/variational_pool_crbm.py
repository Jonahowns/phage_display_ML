from rbm_torch.models.pool_crbm import pool_CRBM

import torch.nn as nn
import torch




class variational_pcrbm(pool_CRBM):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision, meminfo=False)

        assert self.sampling_strategy in ["random", "stratified"]

        # for key in self.hidden_convolution_keys:
        #     self.register_parameter(f"var_mean_{key}", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device)))
        #     self.register_parameter(f"var_std_{key}", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device)))

    def elbo(self, v, h, var_params):
        return self.energy(v, h).mean() - self.entropy(var_params)

    def sample_variational(self, size, variational_param_dict):
        samples = []
        for key in self.hidden_convolution_keys:
            z = torch.normal(0, 1, size=(size, self.convolution_topology[key]["number"],), device=self.device)
            samples.append(variational_param_dict[f"var_mean_{key}"] + z*variational_param_dict[f"var_std_{key}"])
        return samples

    def fit_variational(self, v):
        var_params = torch.nn.ParameterDict({})
        for key in self.hidden_convolution_keys:
            var_params[f"var_mean_{key}"] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            var_params[f"var_std_{key}"] = nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device))

        tmp_optim = torch.optim.SGD(var_params.values(), lr=0.1, momentum=0.2)
        grad_norm = 1.
        while grad_norm > 0.1:  # test for convergence
            htilde = self.sample_variational(v.shape[0], var_params)
            loss = self.elbo(v, htilde, var_params)
            loss.backward()
            grad_norm = self.grad_norm(var_params)
            tmp_optim.step()
            tmp_optim.zero_grad()
        return var_params

    def grad_norm(self, params):
        total_norm = torch.zeros((1,), device=self.device)
        for p in params.values():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5



    def entropy(self, var_params):
        entropy = torch.zeros((1,) , device=self.device)
        for key in self.hidden_convolution_keys:
            entropy_key = 0.5*torch.log(2*torch.pi*torch.square(var_params[f"var_std_{key}"])) + 0.5
            entropy += entropy_key.sum(0)
        return entropy

    def training_step(self, batch, batch_idx):
        seqs, one_hot, seq_weights = batch

        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        vp = self.fit_variational(one_hot)

        # htilde = self.sample_variational(one_hot.shape[0])
        #
        # elbo_loss = -self.elbo(one_hot, htilde)

        loss = vp + reg1 + reg2 + reg3 + bs_loss + gap_loss

        F_v = (self.free_energy(one_hot) * seq_weights / seq_weights.sum()).sum()  # free energy of training data

        logs = {"loss": loss,
                "neg_elbo": elbo_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/neg_elbo_loss", elbo_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs
