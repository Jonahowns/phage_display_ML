from rbm_torch.models.pool_crbm import pool_CRBM
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule, Trainer



class pool_DBM(pool_CRBM):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision, meminfo=False)

        self.pool_h_size = 0
        for key in self.hidden_convolution_keys:
            self.pool_h_size += self.convolution_topology[key]["number"]
            self.register_parameter(f"{key}_M", nn.Parameter(self.weight_intial_amplitude * torch.randn((self.pool_h_size, self.h2_size), device=self.device)))

        self.h2_size = config["h2_size"]

        self.register_parameter(f"h2_fields", nn.Parameter(torch.zeros((self.h2_size), device=self.device)))
        self.register_parameter(f"h2_fields0", nn.Parameter(torch.zeros((self.h2_size), device=self.device)))

    def compute_output_h2(self, h2): # h2*w2.T
        outputs = []
        for key in self.hidden_convolution_keys:
            outputs.append(torch.matmul(h2, getattr(self, f"{key}_M").T))
        return outputs

    def compute_output_h_for_h2(self, h):  # h is list of hus?
        # flatten list
        outputs = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            torch.matmul(h[kid], getattr(self, f"{key}_M"))

        return torch.stack(outputs, dim=1).sum(1)

    def mean_h2(self, psi, beta=1):
        if beta == 1:
            return torch.special.expit(psi + getattr(self, "fields").unsqueeze(0).squeeze(-1))
        else:
            return torch.special.expit(beta * psi + getattr(self, "fields0").unsqueeze(0).squeeze(-1) + beta * (getattr(self, "fields").unsqueeze(0).squeeze(-1) - getattr(self, "fields0").unsqueeze(0).squeeze(-1)))

    def sample_from_inputs_h2(self, psi, beta=1):
        return (torch.randn(psi.shape, device=self.device) < self.mean_h2(psi, beta=beta)).double()

    def logpartition_h2(self, inputs, beta=1):
        if beta == 1:
            return torch.log(1 + torch.exp(getattr(self, "h2_fields").unsqueeze(0).squeeze(-1) + inputs)).sum(1)
        else:
            return torch.log(1 + torch.exp((beta * getattr(self, "h2_fields")).unsqueeze(0).squeeze(-1) + (1 - beta) * getattr(self, "h2_fields0")).unsqueeze(0).squeeze(-1) + beta * inputs).sum(1)

    def markov_step(self, v, h2, beta=1):
        # Gibbs Sampler
        h_input_v = self.compute_output_v_for_h(v)
        h_input_h2 = self.compute_output_h2(h2)
        h_input = [torch.add(h_input_v[i], h_input_h2[i]) for i in range(len(h_input_v))]
        h = self.sample_from_inputs_h(h_input, beta=beta)
        return self.sample_from_inputs_v(self.compute_output_h_for_v(h), beta=beta), h, self.sample_from_inputs_h2(self.compute_output_h_for_h2(h))

    def energy_h2(self, h2, remove_init=False):
        if remove_init:
            return -1. * (h2 * ((getattr(self, "fields") - getattr(self, "fields0")).unsqueeze(0).squeeze(-1)))
        else:
            return -1. * (h2 * getattr(self, "fields").unsqueeze(0).squeeze(-1)).sum((2, 1))

    def energy(self, v, h, h2, remove_init=False, hidden_sub_index=-1):
        return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - \
            self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index) + self.energy_h2(h2, remove_init=remove_init) + self.hidden_coupling_weight_term(h, h2, hidden_sub_index=hidden_sub_index)

    def hidden_coupling_weight_term(self, h, h2, hidden_sub_index=-1):
        # computes h*m term
        if hidden_sub_index != -1:
            h = [subh[hidden_sub_index] for subh in h]

        hm = self.compute_output_h_for_h2(h)  # h * m
        return torch.matmul(hm, h2)

    def energy_PT(self, v, h, h2, N_PT, remove_init=False):
        E = torch.zeros((N_PT, v.shape[1]), device=self.device)
        for i in range(N_PT):
            E[i] = self.energy(v[i], h, h2[i], remove_init=remove_init, hidden_sub_index=i)
        return E


    def forward(self, V_pos_ohe):
        h2_init = self.random_init_config_h2()
        if self.sample_type == "gibbs":
            # Gibbs sampling
            # pytorch lightning handles the device
            with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
                first_v, first_h, first_h2 = self.markov_step(V_pos_ohe, h2_init)
                V_neg = first_v.clone()
                fantasy_h2 = first_h2.clone()
                if self.mc_moves - 2 > 0:
                    for _ in range(self.mc_moves - 2):
                        V_neg, fantasy_h, fantasy_h2 = self.markov_step(V_neg, fantasy_h2)

            V_neg, fantasy_h, fantasy_h2 = self.markov_step(V_neg, fantasy_h2)

            # V_neg, h_neg, V_pos, h_pos
            return V_neg, fantasy_h, fantasy_h2, V_pos_ohe, first_h, first_h2

        elif self.sample_type == "pt":
            # Initialize_PT is called before the forward function is called. Therefore, N_PT will be filled

            # Parallel Tempering
            n_chains = V_pos_ohe.shape[0]

            with torch.no_grad():
                fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
                fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
                fantasy_E = self.energy_PT(fantasy_v, fantasy_h, self.N_PT)

                for _ in range(self.mc_moves - 1):
                    fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
                    self.update_betas(self.N_PT)

            fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
            self.update_betas(self.N_PT)

            # V_neg, h_neg, V_pos, h_pos
            return fantasy_v[0], fantasy_h[0], V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

    def random_init_config_h2(self, custom_size=False, zeros=False):
        if custom_size:
            size = (*custom_size, self.v_num)
        else:
            size = (self.batch_size, self.v_num)

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            ### might need rewriting
            return self.sample_from_inputs_h2(torch.zeros(size, device=self.device).flatten(0, -3), beta=0).reshape(size)



    def markov_PT_and_exchange(self, v, h, h2, e, N_PT):
        for i, beta in zip(torch.arange(N_PT), self.betas):
            v[i], htmp, h2[i] = self.markov_step(v[i], h2[i], beta=beta)
            for hid in range(len(self.hidden_convolution_keys)):
                h[hid][i] = htmp[hid]
            e[i] = self.energy(v[i], h, h2[i], hidden_sub_index=i)

        if self.record_swaps:
            particle_id = torch.arange(N_PT).unsqueeze(1).expand(N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, N_PT - 1, 2):
            proba = torch.exp(betadiff[i] * e[i + 1] - e[i]).minimum(torch.ones_like(e[i]))
            swap = torch.rand(proba.shape[0], device=self.device) < proba
            if i > 0:
                v[i:i + 2, swap] = torch.flip(v[i - 1: i + 1], [0])[:, swap]
                h2[i:i + 2, swap] = torch.flip(h2[i - 1: i + 1], [0])[:, swap]
                for hid in range(len(self.hidden_convolution_keys)):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][i - 1: i + 1], [0])[:, swap]
                # h[i:i + 2, swap] = torch.flip(h[i - 1: i + 1], [0])[:, swap]
                e[i:i + 2, swap] = torch.flip(e[i - 1: i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[i - 1: i + 1], [0])[:, swap]
            else:
                v[i:i + 2, swap] = torch.flip(v[:i + 1], [0])[:, swap]
                h2[i:i + 2, swap] = torch.flip(h2[:i + 1], [0])[:, swap]
                for hid in range(len(self.hidden_convolution_keys)):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][:i + 1], [0])[:, swap]
                e[i:i + 2, swap] = torch.flip(e[:i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[:i + 1], [0])[:, swap]

            if self.record_acceptance:
                self.acceptance_rates[i] = swap.type(torch.get_default_dtype()).mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[
                    i] * (1 - self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps += 1
        return v, h, h2, e

    def AIS(self, M=10, n_betas=10000, batches=None, verbose=0, beta_type='adaptive'):
        with torch.no_grad():
            if beta_type == 'linear':
                betas = torch.arange(n_betas, device=self.device) / torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
            elif beta_type == 'root':
                betas = torch.sqrt(torch.arange(n_betas, device=self.device)) / torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
            elif beta_type == 'adaptive':
                Nthermalize = 200
                Nchains = 20
                N_PT = 11
                self.adaptive_PT_lr = 0.05
                self.adaptive_PT_decay = True
                self.adaptive_PT_lr_decay = 10 ** (-1 / float(Nthermalize))
                if verbose:
                    t = time.time()
                    print('Learning betas...')
                self.gen_data(N_PT=N_PT, Nchains=Nchains, Lchains=1, Nthermalize=Nthermalize, update_betas=True)
                if verbose:
                    print('Elapsed time: %s, Acceptance rates: %s' % (time.time() - t, self.mav_acceptance_rates))
                betas = []
                sparse_betas = self.betas.flip(0)
                for i in range(N_PT - 1):
                    betas += list(sparse_betas[i] + (sparse_betas[i + 1] - sparse_betas[i]) * torch.arange(n_betas / (N_PT - 1), device=self.device) / (n_betas / (N_PT - 1) - 1))
                betas = torch.tensor(betas, device=self.device)
                # if verbose:
                # import matplotlib.pyplot as plt
                # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()

            # Initialization.
            log_weights = torch.zeros(M, device=self.device)
            # config = self.gen_data(Nchains=M, Lchains=1, Nthermalize=0, beta=0)

            config = [self.sample_from_inputs_v(self.random_init_config_v(custom_size=(M,))),
                      self.sample_from_inputs_h(self.random_init_config_h(custom_size=(M,))),
                      self.sample_from_inputs_h2(self.random_init_config_h2(custom_size=(M,)))]

            log_Z_init = torch.zeros(1, device=self.device)

            log_Z_init += self.logpartition_h(self.random_init_config_h(custom_size=(1,), zeros=True), beta=0)
            log_Z_init += self.logpartition_v(self.random_init_config_v(custom_size=(1,), zeros=True), beta=0)
            log_Z_init += self.logpartition_h2(self.random_init_config_h2(custom_size=(1,), zeros=True), beta=0)

            if verbose:
                print(f'Initial evaluation: log(Z) = {log_Z_init.data}')

            for i in range(1, n_betas):
                if verbose:
                    if (i % 2000 == 0):
                        print(f'Iteration {i}, beta: {betas[i]}')
                        print('Current evaluation: log(Z)= %s +- %s' % ((log_Z_init + log_weights).mean(), (log_Z_init + log_weights).std() / np.sqrt(M)))

                config[0], config[1], config[2] = self.markov_step(config[0], config[2])
                energy = self.energy(config[0], config[1], config[2])
                log_weights += -(betas[i] - betas[i - 1]) * energy
            self.log_Z_AIS = (log_Z_init + log_weights).mean()
            self.log_Z_AIS_std = (log_Z_init + log_weights).std() / np.sqrt(M)
            if verbose:
                print('Final evaluation: log(Z)= %s +- %s' % (self.log_Z_AIS, self.log_Z_AIS_std))
            return self.log_Z_AIS, self.log_Z_AIS_std

    def likelihood(self, data, labels, recompute_Z=False):
        if (not hasattr(self, 'log_Z_AIS')) | recompute_Z:
            self.AIS()
        return -((1+self.alpha) * self.free_energy_discriminative(data, labels) + self.alpha*self.free_energy(data)) - self.log_Z_AIS

    def gen_data(self, Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, config_init=[], beta=1, batches=None, reshape=True, record_replica=False, record_acceptance=None, update_betas=None, record_swaps=False):
        """
        Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
        Inputs :
            Nchains (10): Number of Markov chains
            Lchains (100): Length of each chain
            Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
            Nstep (1): Number of Gibbs sampling steps between each sample of a chain
            N_PT (1): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
            batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
            reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
            config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
            beta (1): The inverse temperature of the model.
        """
        with torch.no_grad():
            self.initialize_PT(N_PT, n_chains=Nchains, record_acceptance=record_acceptance, record_swaps=record_swaps)

            if batches == None:
                batches = Nchains
            n_iter = int(Nchains / batches)
            Ndata = Lchains * batches
            if record_replica:
                reshape = False

            if (N_PT > 1):
                if record_acceptance == None:
                    record_acceptance = True

                if update_betas == None:
                    update_betas = False

                # if record_acceptance:
                #     self.mavar_gamma = 0.95

                if update_betas:
                    record_acceptance = True
                    # self.update_betas_lr = 0.1
                    # self.update_betas_lr_decay = 1
            else:
                record_acceptance = False
                update_betas = False

            if (N_PT > 1) and record_replica:
                visible_data = self.random_init_config_v(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                hidden_data = self.random_init_config_h(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                y_data = self.random_init_config_y(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                data = [visible_data, hidden_data, y_data]
            else:
                visible_data = self.random_init_config_v(custom_size=(Nchains, Lchains), zeros=True)
                hidden_data = self.random_init_config_h(custom_size=(Nchains, Lchains), zeros=True)
                y_data = self.random_init_config_y(custom_size=(Nchains, Lchains), zeros=True)
                data = [visible_data, hidden_data, y_data]

            if config_init is not []:
                if type(config_init) == torch.tensor:
                    h_layer = self.random_init_config_h()
                    y_layer = self.random_init_config_y()
                    config_init = [config_init, h_layer, y_layer]

            for i in range(n_iter):
                if config_init == []:
                    config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
                                            record_replica=record_replica, record_acceptance=record_acceptance, update_betas=update_betas, record_swaps=record_swaps)
                else:
                    config_init = [config_init[0][batches * i:batches * (i + 1)], config_init[1][batches * i:batches * (i + 1)]]
                    config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
                                            record_replica=record_replica, config_init=config_init, record_acceptance=record_acceptance,
                                            update_betas=update_betas, record_swaps=record_swaps)

                if (N_PT > 1) & record_replica:
                    data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 2).clone()
                    data[2][batches * i:batches * (i + 1)] = torch.swapaxes(config[2], 0, 2).clone()
                    for hid in range(len(self.hidden_convolution_keys)):
                        data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 2).clone()
                else:
                    data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 1).clone()
                    data[2][batches * i:batches * (i + 1)] = torch.swapaxes(config[2], 0, 1).clone()
                    for hid in range(len(self.hidden_convolution_keys)):
                        data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 1).clone()

            if reshape:
                return [data[0].flatten(0, -3), [hd.flatten(0, -3) for hd in data[1]], data[2]]
            else:
                return data

    def _gen_data(self, Nthermalize, Ndata, Nstep, N_PT=1, batches=1, reshape=True, config_init=[], beta=1, record_replica=False, record_acceptance=True, update_betas=False, record_swaps=False):
        with torch.no_grad():

            if N_PT > 1:
                if update_betas or len(self.betas) != N_PT:
                    self.betas = torch.flip(torch.arange(N_PT) / (N_PT - 1) * beta, [0])

                self.acceptance_rates = torch.zeros(N_PT - 1)
                self.mav_acceptance_rates = torch.zeros(N_PT - 1)

            self.count_swaps = 0
            self.record_swaps = record_swaps

            if self.record_swaps:
                self.particle_id = [torch.arange(N_PT)[:, None].repeat(batches, dim=1)]

            Ndata /= batches
            Ndata = int(Ndata)

            if config_init != []:
                config = config_init
            else:
                if N_PT > 1:
                    config = [self.random_init_config_v(custom_size=(N_PT, batches)), self.random_init_config_h(custom_size=(N_PT, batches)), self.random_init_config_y(custom_size=(N_PT, batches))]
                else:
                    config = [self.random_init_config_v(custom_size=(batches,)), self.random_init_config_h(custom_size=(batches,)), self.random_init_config_y(custom_size=(batches))]

            for _ in range(Nthermalize):
                if N_PT > 1:
                    energy = self.energy_PT(config[0], config[1], config[2], N_PT)
                    config[0], config[1], config[2], energy = self.markov_PT_and_exchange(config[0], config[1], config[2], energy, N_PT)
                    if update_betas:
                        self.update_betas(N_PT, beta=beta)
                else:
                    config[0], config[1], config[2] = self.markov_step(config[0], config[2], beta=beta)

            if N_PT > 1:
                if record_replica:
                    data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0]), config[2].clone().unsqueeze(0)]
                else:
                    data = [config[0][0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0], sub_index=0), config[2][0].clone().unsqueeze(0)]
            else:
                data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0]), config[2].clone().unsqueeze(0)]

            if N_PT > 1:
                if Ndata > 1:
                    if record_replica:
                        data_gen_v = self.random_init_config_v(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_h = self.random_init_config_h(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_y = self.random_init_config_y(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_v[0] = config[0].clone()
                        data_gen_y[0] = config[2].clone()

                        clone = self.clone_h(config[1])
                        for hid in range(len(self.hidden_convolution_keys)):
                            data_gen_h[hid][0] = clone[hid]
                    else:
                        data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
                        data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
                        data_gen_y = self.random_init_config_y(custom_size=(Ndata, batches), zeros=True)
                        data_gen_v[0] = config[0][0].clone()
                        data_gen_y[0] = config[2][0].clone()

                        clone = self.clone_h(config[1], sub_index=0)
                        for hid in range(len(self.hidden_convolution_keys)):
                            data_gen_h[hid][0] = clone[hid]
            else:
                data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
                data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
                data_gen_y = self.random_init_config_y(custom_size=(Ndata, batches), zeros=True)
                data_gen_v[0] = config[0].clone()
                data_gen_y[0] = config[2].clone()

                clone = self.clone_h(config[1])
                for hid in range(len(self.hidden_convolution_keys)):
                    data_gen_h[hid][0] = clone[hid]

            for n in range(Ndata - 1):
                for _ in range(Nstep):
                    if N_PT > 1:
                        energy = self.energy_PT(config[0], config[1], config[2], N_PT)
                        config[0], config[1], config[2], energy = self.markov_PT_and_exchange(config[0], config[1], config[2], energy, N_PT)
                        if update_betas:
                            self.update_betas(N_PT, beta=beta)
                    else:
                        config[0], config[1], config[2] = self.markov_step(config[0], config[2], beta=beta)

                if N_PT > 1 and Ndata > 1:
                    if record_replica:
                        data_gen_v[n + 1] = config[0].clone()
                        data_gen_y[n + 1] = config[2].clone()

                        clone = self.clone_h(config[1])
                        for hid in range(len(self.hidden_convolution_keys)):
                            data_gen_h[hid][n + 1] = clone[hid]

                    else:
                        data_gen_v[n + 1] = config[0][0].clone()
                        data_gen_y[n + 1] = config[2][0].clone()

                        clone = self.clone_h(config[1], sub_index=0)
                        for hid in range(len(self.hidden_convolution_keys)):
                            data_gen_h[hid][n + 1] = clone[hid]

                else:
                    data_gen_v[n + 1] = config[0].clone()
                    data_gen_y[n + 1] = config[2].clone()

                    clone = self.clone_h(config[1])
                    for hid in range(len(self.hidden_convolution_keys)):
                        data_gen_h[hid][n + 1] = clone[hid]

            if Ndata > 1:
                data = [data_gen_v, data_gen_h, data_gen_y]

            if self.record_swaps:
                print('cleaning particle trajectories')
                positions = torch.tensor(self.particle_id)
                invert = torch.zeros([batches, Ndata, N_PT])
                for b in range(batches):
                    for i in range(Ndata):
                        for k in range(N_PT):
                            invert[b, i, k] = torch.nonzero(positions[i, :, b] == k)[0]
                self.particle_id = invert
                self.last_at_zero = torch.zeros([batches, Ndata, N_PT])
                for b in range(batches):
                    for i in range(Ndata):
                        for k in range(N_PT):
                            tmp = torch.nonzero(self.particle_id[b, :i, k] == 0)[0]
                            if len(tmp) > 0:
                                self.last_at_zero[b, i, k] = i - 1 - tmp.max()
                            else:
                                self.last_at_zero[b, i, k] = 1000
                self.last_at_zero[:, 0, 0] = 0

                self.trip_duration = torch.zeros([batches, Ndata])
                for b in range(batches):
                    for i in range(Ndata):
                        self.trip_duration[b, i] = self.last_at_zero[b, i, np.nonzero(invert[b, i, :] == 9)[0]]

            if reshape:
                data[0] = data[0].flatten(0, -3)
                data[1] = [hd.flatten(0, -3) for hd in data[1]]
                data[2] = data[2].flatten(0, -3)
            else:
                data[0] = data[0]
                data[1] = data[1]
                data[2] = data[2]

            return data

