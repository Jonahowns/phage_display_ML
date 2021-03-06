
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
# class Clamp(torch.autograd.Function): # clamp parameter values
#     @staticmethod
#     def forward(ctx, input):
#         return input.clamp(min=0, max=1) # the value in iterative = 2
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clone()

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


# FROM CRBM
# class hidden:
#     def __init__(self, convolution_topology, hidden_keys, datalengths, q):
#         conv_top = {}
#         data = {}
#         for iid, i in enumerate(hidden_keys):
#             conv_top[iid] = convolution_topology[i]
#
#             conv_top[iid]["weight_dims"] = {}
#             conv_top[iid]["conv_dims"] = {}
#             for dl in datalengths:
#                 example_input = (50, 1, dl, q)
#                 dims = conv2d_dim(example_input, conv_top[iid])
#                 conv_top[iid]["weight_dims"][dl] = dims["weight_shape"]
#                 conv_top[iid]["conv_shape"][dl] = dims["conv_shape"]
#                 conv_top[iid]["output_padding"][dl] = dims["output_padding"]
#
#         self.hidden_params = conv_top
#
# rbm_lat = CRBM(config, debug=True)
    # rbm_lat.prepare_data()
    # td = rbm_lat.train_dataloader()
    # for iid, i in enumerate(td):
    #     if iid > 0:
    #         break
    #     seq, cat, ohe, seq_weights = i
    #     rbm_lat(ohe)
    # #     v_out = rbm_lat.compute_output_v(ohe)
    # #     # h = rbm_lat.sample_from_inputs_h(v_out)
    # #     # h_out = rbm_lat.compute_output_h(h)
    # #     # nv = rbm_lat.sample_from_inputs_v(h_out)
    # #     # fe = rbm_lat.free_energy(nv)
    #     pl = rbm_lat.pseudo_likelihood(ohe)
    #     print(pl.mean())

# plt = Trainer(gpus=1, max_epochs=10)
# plt = Trainer(gpus=1, profiler='advanced', max_epochs=10)
# plt = Trainer(profiler='advanced', max_epochs=10)
# plt = Trainer(max_epochs=1)
# plt.fit(rbm)


##### DATA PREP STUFF

# def initial_report(i):
#     seqs = fasta_read(mfolder+rounds[i-1])
#     seqs, cpy_num, df = data_prop(seqs, rounds[i-1], outfile=odir+rounds[i-1]+'seq_len_report.txt')
#     return df
#
#
# def extract_data(i, c_indices, datatypedict, uniform_length=True):
#     seqs = fasta_read(mfolder+rounds[i-1])
#     c_indices = datatypedict["cluster_indices"]
#     cnum = datatypedict["clusters"]
#     seqs, cpy_num, df = data_prop(seqs, f"r{i}", outfile=odir + rounds[i-1] + 'seq_len_report.txt')
#     extractor(seqs, cnum, c_indices, odir + f"r{i}", cpy_num, uniform_length=uniform_length, position_indx=datatypedict["gap_position_indices"])

# Processing the Files
# for j in range(1, len(rounds)+1):
#     df = initial_report(j)
#     # dfs.append(df)
#     for i in range(datatype["clusters"]):
#         extract_data(j, datatype, uniform_length=True)


# def initial_report(i):
#     seqs = fasta_read(mfolder+rounds[i-1])
#     seqs, cpy_num, df = data_prop(seqs, rounds[i-1], outfile=odir+rounds[i-1]+'seq_len_report.txt')
#     return df
#
#
# def extract_data(i, c_indices, datatypedict, uniform_length=True):
#     seqs = fasta_read(mfolder+rounds[i-1])
#     c_indices = datatypedict["cluster_indices"]
#     cnum = datatypedict["clusters"]
#     seqs, cpy_num, df = data_prop(seqs, f"r{i}", outfile=odir + rounds[i-1] + 'seq_len_report.txt')
#     extractor(seqs, cnum, c_indices, odir + f"r{i}", cpy_num, uniform_length=uniform_length, position_indx=datatypedict["gap_position_indices"])

# Processing the Files
# for j in range(1, len(rounds)+1):
#     df = initial_report(j)
#     # dfs.append(df)
#     for i in range(datatype["clusters"]):
#         extract_data(j, datatype, uniform_length=True)






# For switching b/t datasets that were processed differently



#### Prepare Submission Scripts
#
# if focus == "pig":
#     # path is from ProteinMotifRBM/ to /pig/trained_rbms/ or /pig/trained_crbms/
#     dest_path = f"../pig/{datatype_dir}/trained_{model}s/"
#     src_path = f"../pig/{datatype_dir}/"
#
#     all_data_files = []
#     all_model_names = []
#     for i in range(datatype["clusters"]):
#         all_data_files += [x + f'_c{i+1}.fasta' for x in rounds]
#         all_model_names += [x+f"_c{i+1}" for x in rounds]
#
#     script_names = ["pig_" + datatype["id"] +"_"+str(i) for i in range(len(all_model_names))]
#
#     paths_to_data = [src_path + x for x in all_data_files]
#
# elif focus == "invivo":
#     # path is from ProteinMotifRBM/agave_sbumit/ to /pig/trained_rbms/
#     dest_path = f"../invivo/trained_{model}s/"
#     src_path = "../invivo/"
#
#     c1 = [x + '_c1.fasta' for x in rounds]
#     c2 = [x + '_c2.fasta' for x in rounds]
#
#     all_data_files = c1 + c2
#
#     model1 = [x + '_c1' for x in rounds]
#     model2 = [x + '_c2' for x in rounds]
#
#     all_model_names = model1 + model2
#
#     script_names = ["invivo" + str(i) for i in range(len(all_model_names))]
#
#     paths_to_data = [src_path + x for x in all_data_files]
#
#
#
# if focus == "cov":
#     # data type variable
#     datatype_dir = datatype["process"]+f"_{datatype['clusters']}_clusters"
#     # path is from ProteinMotifRBM/ to /pig/trained_rbms/
#     dest_path = f"../cov/{datatype_dir}/trained_{model}s/"
#     src_path = f"../cov/"
#
#     all_data_files = [x + '.fasta' for x in rounds]
#
#     all_model_names = rounds
#
#     script_names = ["cov"+str(i+1) for i in range(len(all_model_names))]
#
#     paths_to_data = [src_path + x for x in all_data_files]


# Processing the Files
# for j in range(1, len(rounds)+1):
    # df = initial_report(j)
#     dfs.append(df)
#     extract_data(j, 1, [[40, 40]])

# def write_submission_scripts(modelnames, script_names, paths_to_data, destination, hiddenunits, focus, epochs, weights=False, gaps=True, gpus=1, partition="htc", pnum=1):
#     # NAME DATA_PATH DESTINATION HIDDEN
#     for i in range(len(modelnames)):
#         if partition == "htc":
#             # Special Subset as it has max wall time strictly at 4 hours
#             o = open(f'rbm_torch/submission_templates/{model}_train_htc.sh', 'r')
#         else:
#             o = open(f'rbm_torch/submission_templates/{model}_train.sh', 'r')
#         filedata = o.read()
#         o.close()
#
#         if "c"+str(1) in modelnames[i]: # cluster 1 has 22 visible units
#             vis = 22
#         elif "c"+str(2) in modelnames[i]:# cluster 2 has 22 visible units
#             vis = 45
#         elif "r" in modelnames[i]:
#             vis = 40 # Cov data is 40 Nucleotides
#
#
#         # Replace the Strings we want
#         filedata = filedata.replace("NAME", modelnames[i]+script_names[i])
#         filedata = filedata.replace("FOCUS", focus)
#         filedata = filedata.replace("DATA_PATH", paths_to_data[i])
#
#         if partition == "sulcgpu":
#             filedata = filedata.replace("PARTITION", f"sulcgpu{pnum}")
#             filedata = filedata.replace("QUEUE", "sulcgpu1")
#         elif partition in ["htc", "htcgpu"] :
#             filedata = filedata.replace("PARTITION", "htcgpu")
#             filedata = filedata.replace("QUEUE", "normal")
#         else:
#             filedata = filedata.replace("PARTITION", f"{partition}{pnum}")
#             filedata = filedata.replace("QUEUE", "wildfire")
#         filedata = filedata.replace("GPU_NUM", str(gpus))
#         filedata = filedata.replace("EPOCHS", str(epochs))
#         filedata = filedata.replace("WEIGHTS", str(weights))
#
#         with open(f"./rbm_torch/agave_submit_{model}/" + script_names[i], 'w+') as file:
#             file.write(filedata)
#
#     if weights:
#         focus += "_w"
#     with open(f"./rbm_torch/agave_submit_{model}/submit" + focus + ".sh", 'w+') as file:
#         file.write("#!/bin/bash\n")
#         for i in range(len(script_names)):
#             file.write("sbatch " + script_names[i] + "\n")
#
#
# write_submission_scripts(all_model_names, script_names, paths_to_data, dest_path, 20, focus, 200, weights=False, gaps=False, gpus=2, partition="sulcgpu2")
# #
# w_script_names = [x+"_w" for x in script_names]
# #
# write_submission_scripts(all_model_names, w_script_names, paths_to_data, dest_path, 20, focus, 200, weights=True, gaps=False, gpus=2, partition="sulcgpu2")

## Alternate sampler for hidden units? Doesn't work as the truncated normal from pytorch only works for 1 set value, not a matrix of means/ std devs
# def sample_from_inputs_h(self, psi, nancheck=False, beta=1):  # psi is a list of hidden Iuks
    #     h_uks = []
    #     for iid, i in enumerate(self.hidden_convolution_keys):
    #         if beta == 1:
    #             a_plus = getattr(self, f'{i}_gamma+').unsqueeze(0).unsqueeze(2)
    #             a_minus = getattr(self, f'{i}_gamma-').unsqueeze(0).unsqueeze(2)
    #             theta_plus = getattr(self, f'{i}_theta+').unsqueeze(0).unsqueeze(2)
    #             theta_minus = getattr(self, f'{i}_theta-').unsqueeze(0).unsqueeze(2)
    #         else:
    #             theta_plus = (beta * getattr(self, f'{i}_theta+') + (1 - beta) * getattr(self, f'{i}_0theta+')).unsqueeze(0).unsqueeze(2)
    #             theta_minus = (beta * getattr(self, f'{i}_theta-') + (1 - beta) * getattr(self, f'{i}_0theta-')).unsqueeze(0).unsqueeze(2)
    #             a_plus = (beta * getattr(self, f'{i}_gamma+') + (1 - beta) * getattr(self, f'{i}_0gamma+')).unsqueeze(0).unsqueeze(2)
    #             a_minus = (beta * getattr(self, f'{i}_gamma-') + (1 - beta) * getattr(self, f'{i}_0gamma-')).unsqueeze(0).unsqueeze(2)
    #             psi[iid] *= beta
    #
    #         if nancheck:
    #             nans = torch.isnan(psi[iid])
    #             if nans.max():
    #                 nan_unit = torch.nonzero(nans.max(0))[0]
    #                 print('NAN IN INPUT')
    #                 print('Hidden units', nan_unit)
    #
    #         psi_plus = (-psi[iid]).add(theta_plus).div(torch.sqrt(a_plus))
    #         psi_minus = psi[iid].add(theta_minus).div(torch.sqrt(a_minus))
    #
    #         etg_plus = self.erf_times_gauss(psi_plus)  # Z+ * sqrt(a+)
    #         etg_minus = self.erf_times_gauss(psi_minus)  # Z- * sqrt(a-)
    #
    #         p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
    #         nans = torch.isnan(p_plus)
    #
    #         if True in nans:
    #             p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
    #         p_minus = 1 - p_plus
    #
    #         huk_plus = torch.zeros_like(psi_plus, device=self.device)
    #         huk_minus = torch.zeros_like(psi_minus, device=self.device)
    #
    #         max_val = 1e9
    #
    #         nn.init.trunc_normal_(huk_plus, (psi[iid]-theta_plus)/a_plus, 1/a_plus, 0, max_val)
    #         nn.init.trunc_normal_(huk_minus, (psi[iid]-theta_minus)/a_minus, 1/a_minus, 0, -max_val)
    #
    #         huk = p_plus*huk_plus + p_minus*huk_minus
    #         h_uks.append(huk)
    #     return h_uks


# class BinaryClassifier(nn.module):
#     def __init__(self, crbm_config, dataset="mnist"):
#         super().__init__()
#         self.binary_crbm = BinaryCRBM(crbm_config)
#         self.crbm_epochs = crbm_config["epochs"]
#         self.classifier_epochs = crbm_config["classifier_epochs"]
#
#
#
#
#
#
#     def get_test_rbm_output(self, input_size):
#         return self.binary_crbm.compute_output_v(torch.rand(input_size, device=self.device))
#
#     def train(self, gpus=0, logger_dir="./tb_logs/", name="mnist_crbm"):
#         # First we train the crbm
#         print("Training CRBM")
#         logger = TensorBoardLogger(logger_dir, name=name)
#         plt = Trainer(max_epochs=self.crbm_epochs, logger=logger, gpus=gpus, accelerator="ddp")  # distributed data-parallel
#         plt.fit(self.binary_crbm)
#         # Now we train the classifier
#         print("Training classifier")
#
#
#
#
#
#     def classifier_training_routine(self):
#         td = self.binary_crbm.train_dataloader()
#         vd = self.binary_crbm.val_dataloader()
#
#         self.binary_crbm.eval()  # set to eval so we don't change variables of the binary crbm
#         for _ in range(self.classifier_epochs):
#             for b, b_idx in td: # each training epoch
#                 x, y = b  # images, labels = batch
#                 probs = self(x)
#                 preds = probs.argmax(1)
#                 train_loss = F.cross_entropy(preds, y)
#
#             train_batch_loss = []
#             for vb, vb_idx in vd: # each validation epoch
#                 v, vy = vb
#                 probs = self(v)
#                 preds = probs.argmax(1)
#                 val_loss = F.cross_entropy(preds, vy)
#
#             epoch_accuracy = torchmetrics.Accuracy(p)
#
