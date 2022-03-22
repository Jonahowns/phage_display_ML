
# import math
# from numbers import Number
#
# import torch
# from torch.distributions import Distribution, constraints
# from torch.distributions.utils import broadcast_all
#
# CONST_SQRT_2 = math.sqrt(2)
# CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
# CONST_INV_SQRT_2 = 1 / math.sqrt(2)
# CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
# CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)
#
# # taken from https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
# class TruncatedStandardNormal(Distribution):
#     """
#     Truncated Standard Normal distribution
#     https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     """
#
#     arg_constraints = {
#         'a': constraints.real,
#         'b': constraints.real,
#     }
#     has_rsample = True
#
#     def __init__(self, a, b, validate_args=None):
#         self.a, self.b = broadcast_all(a, b)
#         if isinstance(a, Number) and isinstance(b, Number):
#             batch_shape = torch.Size()
#         else:
#             batch_shape = self.a.size()
#         super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
#         if self.a.dtype != self.b.dtype:
#             raise ValueError('Truncation bounds types are different')
#         if any((self.a >= self.b).view(-1,).tolist()):
#             raise ValueError('Incorrect truncation range')
#         eps = torch.finfo(self.a.dtype).eps
#         self._dtype_min_gt_0 = eps
#         self._dtype_max_lt_1 = 1 - eps
#         self._little_phi_a = self._little_phi(self.a)
#         self._little_phi_b = self._little_phi(self.b)
#         self._big_phi_a = self._big_phi(self.a)
#         self._big_phi_b = self._big_phi(self.b)
#         self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
#         self._log_Z = self._Z.log()
#         little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
#         little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
#         self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
#         self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
#         self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
#         self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z
#
#     @constraints.dependent_property
#     def support(self):
#         return constraints.interval(self.a, self.b)
#
#     @property
#     def mean(self):
#         return self._mean
#
#     @property
#     def variance(self):
#         return self._variance
#
#     @property
#     def entropy(self):
#         return self._entropy
#
#     @property
#     def auc(self):
#         return self._Z
#
#     @staticmethod
#     def _little_phi(x):
#         return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI
#
#     @staticmethod
#     def _big_phi(x):
#         return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
#
#     @staticmethod
#     def _inv_big_phi(x):
#         return CONST_SQRT_2 * (2 * x - 1).erfinv()
#
#     def cdf(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)
#
#     def icdf(self, value):
#         return self._inv_big_phi(self._big_phi_a + value * self._Z)
#
#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5
#
#     def rsample(self, sample_shape=torch.Size()):
#         shape = self._extended_shape(sample_shape)
#         p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
#         return self.icdf(p)
#
#
# class TruncatedNormal(TruncatedStandardNormal):
#     """
#     Truncated Normal distribution
#     https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     """
#
#     has_rsample = True
#
#     def __init__(self, loc, scale, a, b, validate_args=None):
#         self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
#         a = (a - self.loc) / self.scale
#         b = (b - self.loc) / self.scale
#         super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
#         self._log_scale = self.scale.log()
#         self._mean = self._mean * self.scale + self.loc
#         self._variance = self._variance * self.scale ** 2
#         self._entropy += self._log_scale
#
#     def _to_std_rv(self, value):
#         return (value - self.loc) / self.scale
#
#     def _from_std_rv(self, value):
#         return value * self.scale + self.loc
#
#     def cdf(self, value):
#         return super(TruncatedNormal, self).cdf(self._to_std_rv(value))
#
#     def icdf(self, value):
#         return self._from_std_rv(super(TruncatedNormal, self).icdf(value))
#
#     def log_prob(self, value):
#         return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale




#### From RBM_Categorical
# def one_hot(self, seq):
#     one_hot_vector = np.zeros((self.max_length, self.n_bases), dtype=np.float32)
#     for n, base in enumerate(seq):
#         one_hot_vector[n][self.base_to_id[base]] = 1
#     return one_hot_vector.reshape((1, 1, self.n_bases, self.max_length))



# this didn't work
# # enforces a zero sum gauge on weights passed
# class zero_sum(nn.Module):
#     def __init__(self, q):
#         super(zero_sum, self).__init__()
#         self.q = q
#     def forward(self, X):
#         X = X - X.sum(-1).unsqueeze(2) / self.q
#         return X

# may be used to clip parameter values, have yet to use
class Clamp(torch.autograd.Function): # clamp parameter values
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=1) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

######### Loss Debugging
 # psuedo likelihood actually minimized, loss still rises, 1/this causes nan in parameter
# cd_loss = torch.log(1+torch.exp((self.energy(V_pos, h_pos) - self.energy(V_neg, h_neg)).mean()))
# moved the mean operation, this is the best one yet
# cd_loss = torch.log(1+torch.exp(self.energy(V_pos, h_pos).mean() - self.energy(V_neg, h_neg).mean()))

 # minimizes loss, psuedolikelihood goes up
        # cd_loss = torch.log(1+torch.exp(-self.energy(V_pos, h_pos).mean() + self.energy(V_neg, h_neg).mean()))
        # let's try free energy version of this, it
        # cd_loss = torch.log(1+torch.exp((self.free_energy(V_pos) - self.free_energy(V_neg)).mean()))
        # cd_loss = torch.log(1+torch.exp((self.free_energy(V_pos).mean() - self.free_energy(V_neg).mean())))
        # free energy version, loss goes up, slight decrease of psuedolikelihood
        # cd_loss = torch.log(1+torch.exp((self.free_energy(V_pos).mean() - self.free_energy(V_neg).mean())))
        # loss still rises, making this term the denominator


        # flipped version of above, did not work
        # cd_loss = torch.log(1+torch.exp((-self.energy(V_pos, h_pos) + self.energy(V_neg, h_neg)).mean()))
        # energy_p = self.energy(V_pos, h_pos)
        # free_energy_n = self.free_energy(V_neg)
        # free_energy_p = self.free_energy(V_pos)

        # n_term = torch.log(torch.exp(-free_energy_n.mean()))
        # cd_loss = free_energy_n.mean() - free_energy_p.mean()

        # this is what it should be I think
        # cd_loss = free_energy_p.mean() - free_energy_n.mean()

        # cd_loss = free_energy_p.mean() + free_energy_n.mean()
        # cd_loss = free_energy_p.mean() + n_term
        # cd_loss = torch.exp(free_energy_p.mean() + free_energy_n.mean())
        # cd_loss = torch.exp(free_energy_n.mean() - free_energy_p.mean())

        # nll = (energy_p - free_energy_n).mean()

        #
        # energy_pos = self.energy(V_pos, h_pos)  # energy of training data
        # energy_neg = self.energy(V_neg, h_neg)  # energy of gibbs sampled visible states
        # cd_loss = torch.mean(energy_pos) - torch.mean(energy_neg)
        # cd_loss = -torch.mean(energy_pos) + torch.mean(energy_neg)

        # p_vpos =
        # p_vneg =
# E_loss = torch.log(1+torch.exp(dF)) # log loss functional
# E_loss2 = -torch.log(dF)




 # Looking for better Interpretation of weights with potential
    # def energy_per_state(self):
        # inputs 21 x v_num
        # inputs = torch.arange(self.q).unsqueeze(1).expand(-1, self.v_num)
        #
        #
        # indexTensor = inputs.unsqueeze(1).unsqueeze(-1).expand(-1, self.h_num, -1, -1)
        # expandedweights = self.W.unsqueeze(0).expand(inputs.shape[0], -1, -1, -1)
        # output = torch.gather(expandedweights, 3, indexTensor).squeeze(3)
        # out = torch.swapaxes(output, 1, 2)
        # energy = torch.zeros((self.q, self.v_num, self.h_num))
        # for i in range(self.q):
        #     for j in range(self.v_num):
        #         energy[i, j, :] = self.logpartition_h(out[i, j, :])
        #
        # # Iu_flat = output.reshape((self.q*self.h_num, self.v_num))
        # # Iu = self.compute_output_v(inputs)
        #
        # e_h = F.normalize(energy, dim=0)
        # view = torch.swapaxes(e_h, 0, 2)
        #
        # W = self.get_param("W")
        #
        # rbm_utils.Sequence_logo_all(W, name="allweights" + '.pdf', nrows=5, ncols=1, figsize=(10,5) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule="protein")
        # rbm_utils.Sequence_logo_all(view.detach(), name="energything" + '.pdf', nrows=5, ncols=1, figsize=(10,5) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule="protein")

    # def compute_output_v(self, visible_data):
    #     # output = torch.zeros((visible_data.shape[0], self.h_num), device=self.device)
    #
    #     # compute_output of visible potts layer
    #     vd = visible_data.long()
    #
    #     # Newest Version also works, fastest version
    #     indexTensor = vd.unsqueeze(1).unsqueeze(-1).expand(-1, self.h_num, -1, -1)
    #     expandedweights = self.W.unsqueeze(0).expand(visible_data.shape[0], -1, -1, -1)
    #     output = torch.gather(expandedweights, 3, indexTensor).squeeze(3).sum(2)
    #
    #     # vd shape batch_size x visible
    #     # output shape batch size x hidden
    #     # Weight shape hidden x visible x q
    #
    #     # 2nd fastest this works
    #     # for u in range(self.h_num):
    #     #     weight_view = self.W[u].expand(vd.shape[0], -1, -1)
    #     #     output[:, u] += torch.gather(weight_view, 2, vd.unsqueeze(2)).sum(1).squeeze(1)
    #
    #     # previous implementation
    #     # for u in range(self.h_num):  # for u in h_num
    #     #     for v in range(self.v_num):  # for v in v_num
    #     #         output1[:, u] += self.W[u, v, vd[:, v]]
    #
    #     return output

 ## Gradient Clipping for poor behavior, have no need for it yet
    # def on_after_backward(self):
    #     self.grad_norm_clip_value = 10
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip_value)

  ## For debugging of main functions
    # def sampling_test(self):
    #     self.prepare_data()
    #     train_reader = RBMCaterogical(self.training_data, weights=self.weights, max_length=self.v_num, shuffle=False, base_to_id=self.base_to_id, device=self.device)
    #
    #     # initialize fields from data
    #     with torch.no_grad():
    #         initial_fields = train_reader.field_init()
    #         self.params['fields'] += initial_fields
    #         self.params['fields0'] += initial_fields
    #
    #     self.W = self.params['W_raw'] - self.params['W_raw'].sum(-1).unsqueeze(2) / self.q
    #
    #     v = self.random_init_config_v()
    #     h = self.sample_from_inputs_h(self.compute_output_v(v))
    #     v2 = self.sample_from_inputs_v(self.compute_output_h(h))



  # check weights
    # version = 0
    # # checkpoint = get_checkpoint(version, dir="./tb_logs/lattice_trial/")
    # checkpoint = "/mnt/D1/globus/rbm_hyperparam_results/train_rbm_ad5d7_00005_5_l1_2=0.3832,lf=0.00011058,lr=0.065775,weight_decay=0.086939_2022-01-18_11-02-53/checkpoints/epoch=99-step=499.ckpt"
    # rbm = RBM.load_from_checkpoint(checkpoint)
    # rbm.energy_per_state()
    # all_weights(rbm, "./lattice_proteins_verification" + "/allweights", 5, 1, 10, 2, molecule="protein")

    # checkpoint = torch.load(checkpoint_file)
    # model.prepare_data()
    # model.criterion.weight = torch.tensor([0., 0.]) # need to add as this is saved by the checkpoint file
    # model.load_state_dict(checkpoint['state_dict'])



    ## Need to finish debugging AIS
    # rbm = RBM(config)
    # rbm.sampling_test()
    # rbm.AIS()

    # rbm.prepare_data()


    # d = iter(rbm.train_dataloader())
    # seqs, v_pos, weights = d.next()
    # logger.experiment.add_graph(rbm, v_pos)

    # profiler = torch.profiler.profile(profile_memory=True)
    # profiler = pytorch_lightning.profiler.SimpleProfiler(profile_memory=True)
    # profiler = pytorch_lightning.profiler.PyTorchProfiler(profile_memory=True)

    # logger = TensorBoardLogger('tb_logs', name='bench_trial')
    # # plt = Trainer(max_epochs=epochs, logger=logger, gpus=0, profiler=profiler)  # gpus=1,
    # plt = Trainer(max_epochs=epochs, logger=logger, gpus=0, profiler="advanced")  # gpus=1,
    # # tic = time.perf_counter()
    # plt.fit(rbm)
    # toc = time.perf_counter()
    # tim = toc-tic
    #
    # print("Trial took", tim, "seconds")



    # version 11 of fixed trial is 50 epochs of pt sampled


    # check weights
    # version = 15
    # # checkpoint = get_checkpoint(version, dir="./tb_logs/trial/")
    # checkpoint = get_checkpoint(version, dir="./tb_logs/bench_trial/")
    #
    # rbm = RBM.load_from_checkpoint(checkpoint)
    # all_weights(rbm, "./tb_logs/bench_trial/version_" + str(version) + "/allweights", 5, 1, 10, 2, molecule="protein")




    # plt = Trainer(gpus=1, max_epochs=10)
    # plt = Trainer(gpus=1, profiler='advanced', max_epochs=10)
    # plt = Trainer(profiler='advanced', max_epochs=10)
    # plt = Trainer(max_epochs=1)
    # plt.fit(rbm)


    # total = 0
    # for i, batch in enumerate(d):
    #     print(len(batch))
    #     seqs, tens = batch
    #     if i == 0:
    #         # rbm.testing()
    #         rbm.training_step(batch, i)
