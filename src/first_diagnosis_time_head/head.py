"""A separate censored waiting-time likelihood head for each frozen backbone."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeHead(nn.Module):
    def __init__(self, features, durations, observed):
        super().__init__()
        self.register_buffer("center", features.mean(0))
        self.register_buffer("scale", features.std(0, unbiased=False).clamp_min(1e-5))
        self.network = nn.Sequential(nn.Linear(features.shape[1], 64), nn.GELU(), nn.Linear(64, 2))
        target = torch.log1p(durations[observed])
        mu = float(target.median()) if len(target) else math.log1p(365.25)
        sigma = max(.5, min(3., float(target.std(unbiased=False)))) if len(target) else 1.
        with torch.no_grad():
            self.network[-1].weight.mul_(.01)
            self.network[-1].bias.copy_(torch.tensor([mu, math.log(math.expm1(sigma-.05))]))

    def forward(self, features):
        output = self.network((features-self.center) / self.scale)
        return output[:, 0].double(), (.05 + F.softplus(output[:, 1])).clamp_max(5.).double()


def nll(mu, sigma, duration, observed, eligible):
    mu, sigma = mu.double(), sigma.double()
    duration = duration.double()
    z = (torch.log1p(duration)-mu) / sigma
    positive = .5*z.square() + torch.log(sigma) + .5*math.log(2*math.pi) + torch.log1p(duration)
    zero = -torch.special.log_ndtr(-mu/sigma)
    censored = -torch.special.log_ndtr(-z)
    values = torch.where(observed, torch.where(duration == 0, zero, positive), censored)
    return values[eligible].mean()
