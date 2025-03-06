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