
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