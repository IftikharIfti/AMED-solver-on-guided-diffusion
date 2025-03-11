import torch
from torch_utils import persistence
from torch.nn.functional import silu

@persistence.persistent_class
class CGPrecond(torch.nn.Module):
    def __init__(self, model, classifier, guidance_rate=1.0, use_fp16=False):
        super().__init__()
        self.img_resolution = model.image_size
        self.img_channels = model.in_channels
        self.label_dim = model.num_classes
        self.beta_d = 19.9
        self.beta_min = 0.1
        self.M = 1000
        self.epsilon_t = 1e-3
        self.sigma_min = float(self.sigma(self.epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = model
        self.classifier = classifier
        self.guidance_rate = guidance_rate
        self.use_fp16 = use_fp16

    def forward(self, x, sigma, class_labels=None, force_fp32=False, y=None):
        sigma = sigma.reshape(-1, 1, 1, 1)
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), y=class_labels)
        F_x, _ = torch.split(F_x, 3, dim=1)  # Split off variance if present
        if class_labels is not None and self.guidance_rate > 0:
            F_x = self.condition_score(self.cond_fn, F_x, c_in * x, c_noise.flatten(), sigma, y=class_labels)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x.clamp(-1, 1)

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def cond_fn(self, x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.guidance_rate

    def condition_score(self, cond_fn, eps, x, t, sigma, y=None):
        alpha_bar = 1 / (1 + sigma ** 2)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, y=y)
        return eps
# @persistence.persistent_class
class CFGPrecond(torch.nn.Module):
    def __init__(self,
        model,
        guidance_type   = 'classifier-free',
        guidance_rate   = 1.0,
        epsilon_t       = 1e-3,                 # Minimum t-value used during training.
        beta_d          = 9.0420,               # Extent of the noise level schedule.
        beta_min        = 0.8477,               # Initial slope of the noise level schedule.
        img_resolution  = 64,
        img_channels    = 4,
        label_dim       = True,                 # Number of class labels, 0 = unconditional.
        model_type      = 'CFGUNet',            # Class name of the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.model = model
        self.guidance_rate = guidance_rate
        self.guidance_type = guidance_type
        if self.guidance_type == 'classifier-free':
            self.wrapper_fn = lambda x, t, c: self.model.apply_model(x, t, c)
        else:
            self.wrapper_fn = lambda x, t: self.model.apply_model(x, t, None)
        
        alphas_cumprod = model.alphas_cumprod
        log_alphas = 0.5 * torch.log(alphas_cumprod)
        self.M = len(log_alphas)
        self.t_array = torch.linspace(0., 1., self.M + 1)[1:].reshape((1, -1))
        self.log_alpha_array = log_alphas.reshape((1, -1,))

        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        
    def noise_pred_fn(self, x, c_noise, cond=None):
        if c_noise.reshape((-1,)).shape[0] == 1:
            c_noise = c_noise.expand((x.shape[0]))
        t_input = c_noise
        if cond is None:
            output = self.wrapper_fn(x, t_input)
        else:
            output = self.wrapper_fn(x, t_input, cond)
        return output
        
    def forward(self, x, sigma, condition=None, unconditional_condition=None, **model_kwargs):
        sigma = sigma.reshape(-1,)

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M * self.sigma_inv(sigma) - 1.

        if c_noise.reshape((-1,)).shape[0] == 1:
            c_noise = c_noise.expand((x.shape[0]))
        if self.guidance_type == "uncond":
            F_x = self.noise_pred_fn(c_in.reshape(-1,1,1,1) * x, c_noise)
        elif self.guidance_type == "classifier-free":
            if self.guidance_rate == 1. or unconditional_condition is None:
                F_x = self.noise_pred_fn(c_in * x, c_noise, cond=condition)
            else:
                x_in = torch.cat([c_in.reshape(-1,1,1,1) * x] * 2)
                t_in = torch.cat([c_noise] * 2)
                cond_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = self.noise_pred_fn(x_in, t_in, cond=cond_in).chunk(2)
                F_x = noise_uncond + self.guidance_rate * (noise - noise_uncond)

        D_x = c_skip * x + c_out.reshape(-1,1,1,1) * F_x

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def marginal_log_mean_coeff(self, t):
        t = torch.tensor(t)
        return self.interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def sigma_inv(self, sigma):
        lamb = -(sigma.log())
        log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
        t = self.interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
        return t.reshape((-1,))
    
    def interpolate_fn(self, x, xp, yp):
        """
        A piecewise linear function y = f(x), using xp and yp as keypoints.
        We implement f(x) in a differentiable way (i.e. applicable for autograd).
        The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

        Args:
            x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
            xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
            yp: PyTorch tensor with shape [C, K].
        Returns:
            The function values f(x), with shape [N, C].
        """
        N, K = x.shape[0], xp.shape[1]
        all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
        sorted_all_x, x_indices = torch.sort(all_x, dim=2)
        x_idx = torch.argmin(x_indices, dim=2)
        cand_start_idx = x_idx - 1
        start_idx = torch.where(
            torch.eq(x_idx, 0),
            torch.tensor(1, device=x.device),
            torch.where(
                torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
            ),
        )
        end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
        start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
        end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
        start_idx2 = torch.where(
            torch.eq(x_idx, 0),
            torch.tensor(0, device=x.device),
            torch.where(
                torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
            ),
        )
        y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
        start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
        end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
        cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
        return cand
