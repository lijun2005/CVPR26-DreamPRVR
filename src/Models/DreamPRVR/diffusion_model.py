import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import math
import copy

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.true_divide(torch.arange(0, d_model, step=2), d_model) * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BottleNeck(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.blcok_in = nn.Sequential(
            nn.LayerNorm(input_dim),
            Swish(),
            nn.Linear(input_dim, hidden_dim),
        )
        self.short_cut = nn.Linear(output_dim,output_dim)

        self.blcok_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, t, cond):
        h = self.blcok_in(x)
        h += self.temb_proj(t.unsqueeze(1))
        h += self.cond_proj(cond)
        h = self.blcok_out(h)
        h = h + self.short_cut(x)
        return h

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, T=100, dropout=0.0):
        super().__init__()

        self.layers = _get_clones(BottleNeck(input_dim, hidden_dim, output_dim, dropout), num_layers)

        self.time_embedding = TimeEmbedding(T, input_dim, hidden_dim)

        self.cond_embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
        )

        self.short_cut = nn.Linear(input_dim, output_dim)
    def forward(self, x, t, cond):

        temb = self.time_embedding(t)
        cond = self.cond_embedding(cond)
        h = x
        for _, layer in enumerate(self.layers):
            h = layer(h, temb, cond)

        x = self.short_cut(h + x)

        return x


# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
            self,
            timesteps=1000,
            beta_schedule='cosine'#
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.timesteps = timesteps

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, cond, t):
        # predict noise using model
        pred_noise = model(x_t, t, cond)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    # @torch.no_grad()
    def p_sample(self, model, x_t, cond, t, middle_noise=None):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, cond, t)
        noise = torch.randn_like(x_t) if middle_noise == None else middle_noise
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    # @torch.no_grad()
    def p_sample_loop(self, model, cond, shape, noise=None, middle_noise=None):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device) if noise == None else noise
        for i in reversed(range(0, self.timesteps)):
            # middle_noise = middle_noise[i] if middle_noise !=None else None
            img = self.p_sample(model, img, cond, torch.full((batch_size,), i, device=device, dtype=torch.long),
                                 middle_noise=middle_noise)
        return img

    # sample new images
    # @torch.no_grad()
    def sample(self, model, cond=None, noise=None, middle_noise=None):
        batchsize,numregister, ch_dim = cond.size()
        return self.p_sample_loop(model, cond, shape=(batchsize,numregister, ch_dim), noise=noise, middle_noise=middle_noise)

    # sample more timesteps for each iteration for faster convergence
    def ddpm_forward_oversample(self, model, x_start, cond, noise=None,type=0):
        device = next(model.parameters()).device
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device=device).long()
        # generate random noise
        noise = torch.randn_like(x_start) if noise==None else noise
        predicted_noise_list = []
        noise_list = []
        x_recon_list = []
        x_noisy_list = []
        for i in range(self.timesteps):
            t = i * torch.ones((x_start.size(0),), device=device).long()
            # get x_t
            x_noisy = self.q_sample(x_start, t, noise=noise)
            predicted_noise = model(x_noisy, t, cond)
            x_recon = self.predict_start_from_noise(x_noisy, t, predicted_noise)

            predicted_noise_list.append(predicted_noise)
            noise_list.append(noise)
            x_recon_list.append(x_recon)
            x_noisy_list.append(x_noisy)
        # Shape will be (T, B, K, D)
        all_predicted_noise = torch.stack(predicted_noise_list, dim=0)
        

        all_original_noise = noise.unsqueeze(0).expand(self.timesteps, -1, -1, -1)

        all_x_recon_list = torch.stack(x_recon_list, dim=0)
        all_x_noisy_list = torch.stack(x_noisy_list, dim=0)

        return all_original_noise, all_predicted_noise,all_x_recon_list,all_x_noisy_list         

