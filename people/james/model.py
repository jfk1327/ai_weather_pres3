"""Convolutional VAE for 32x32 daily-max wind maps, and its loss.

Input:  x of shape (B, 1, 32, 32), z-scored (see dataset.py), so values are unbounded, roughly -2 to +7.
Latent: z of shape (B, latent_dim), prior p(z) = N(0, I).
Output: reconstruction of shape (B, 1, 32, 32), in the same z-scored space as x.

Self-test (run after implementing):
    python people/james/model.py

Docs:
  - torch.nn.Conv2d, section "Shape" (the H_out formula: work out 32 -> ? for your kernel/stride/padding)
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
  - torch.nn.ConvTranspose2d, section "Shape" (and the note on output_padding)
    https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
  - torch.nn.Linear, torch.flatten / Tensor.view, torch.randn_like
  - Kingma & Welling 2013, "Auto-Encoding Variational Bayes", Section 2.4 (reparameterisation trick)
    and Appendix B (closed-form KL between N(mu, sigma^2) and N(0, 1))
    https://arxiv.org/abs/1312.6114
"""
import time

import torch
from torch import nn


# TODO(human): implement the four pieces below (marked TODO). Decide the architecture and the
# loss reduction, and justify both in a comment.
# Weigh: how 32x32 shrinks to a small bottleneck (write the shape after every layer as a comment);
# the decoder must exactly mirror back to (B, 1, 32, 32); x is z-scored, so what output activation
# (if any) fits? The loss must not change meaning with batch size: sum or mean, over what?

class Encoder(nn.Module):
    """x (B, 1, 32, 32) -> (mu, logvar), each (B, latent_dim).

    TODO: conv layers that downsample, then flatten, then two linear heads (one for mu, one for logvar).
    Why predict log-variance rather than the standard deviation directly?
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Decoder(nn.Module):
    """z (B, latent_dim) -> reconstruction (B, 1, 32, 32).

    TODO: linear layer up to a small feature map, reshape, then transposed convs that upsample to 32x32.
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        raise NotImplementedError

    def forward(self, z):
        raise NotImplementedError


class VAE(nn.Module):
    """Encoder + reparameterisation + decoder.

    forward(x) returns (recon, mu, logvar).
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterise(self, mu, logvar):
        """Sample z ~ N(mu, sigma^2) so that gradients flow back into mu and logvar.

        TODO: why can't you just call torch.normal(mu, sigma) here? What does the trick move
        the randomness into?
        """
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Negative ELBO with a beta weight on the KL term.

    Args:
        recon, x: (B, 1, 32, 32)
        mu, logvar: (B, latent_dim)
    Returns:
        (total, recon_term, kl_term): three scalar tensors, total = recon_term + beta * kl_term.

    TODO: reconstruction = squared error (which likelihood does MSE correspond to?);
    KL = closed form KL(N(mu, sigma^2) || N(0, I)), summed over latent dims.
    Decide the reduction over pixels and over the batch, and keep it consistent between the two terms:
    changing it silently changes what beta means.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Self-test: checks the contract train.py will rely on. Don't edit; make it pass.
    torch.manual_seed(0)
    B, D = 8, 16
    model = VAE(latent_dim=D)
    x = torch.randn(B, 1, 32, 32)

    mu, logvar = model.encoder(x)
    assert mu.shape == (B, D) and logvar.shape == (B, D), f"encoder gave {tuple(mu.shape)}, {tuple(logvar.shape)}"
    out = model.decoder(torch.randn(B, D))
    assert out.shape == (B, 1, 32, 32), f"decoder gave {tuple(out.shape)}, expected ({B}, 1, 32, 32)"

    recon, mu, logvar = model(x)
    assert recon.shape == x.shape, f"forward recon shape {tuple(recon.shape)}"

    # reparameterisation: tiny variance -> z is (almost) mu; and it must be random otherwise
    z_tight = model.reparameterise(mu, torch.full_like(logvar, -30.0))
    assert torch.allclose(z_tight, mu, atol=1e-5), "with logvar=-30, z should equal mu"
    zero = torch.zeros(B, D)
    assert not torch.equal(model.reparameterise(zero, zero), model.reparameterise(zero, zero)), \
        "reparameterise should be random"

    # loss: zero error and zero KL in the right places, scalars, non-negative
    total, rec, kl = vae_loss(x, x, zero, zero)
    assert total.dim() == rec.dim() == kl.dim() == 0, "loss terms must be scalars"
    assert rec.abs() < 1e-6, "recon term should be 0 when recon == x"
    assert kl.abs() < 1e-6, "KL should be 0 when mu=0, logvar=0 (posterior == prior)"
    _, _, kl_pos = vae_loss(x, x, torch.ones(B, D), zero)
    assert kl_pos > 0, "KL should be > 0 when mu != 0"
    t2, r2, k2 = vae_loss(recon, x, mu, logvar, beta=2.0)
    assert torch.allclose(t2, r2 + 2.0 * k2), "total should be recon + beta * kl"

    # batch-size invariance: the same sample repeated should give the same loss for B=1 and B=8
    t1, _, _ = vae_loss(recon[:1].expand(B, -1, -1, -1), x[:1].expand(B, -1, -1, -1),
                        mu[:1].expand(B, -1), logvar[:1].expand(B, -1))
    t0, _, _ = vae_loss(recon[:1], x[:1], mu[:1], logvar[:1])
    assert torch.allclose(t0, t1, rtol=1e-4), "loss changes with batch size: average over the batch"

    # gradients reach every parameter
    total, _, _ = vae_loss(recon, x, mu, logvar)
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} ({n_params * 4 / 2**20:.1f} MB as float32)")

    # cost on real data: one epoch's worth of forward + backward passes
    from dataset import make_loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, _, _, _ = make_loaders("gust", batch_size=128)
    model = VAE(latent_dim=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t = time.time()
    for xb in train_dl:
        xb = xb.to(device)
        recon, mu, logvar = model(xb)
        loss, _, _ = vae_loss(recon, xb, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"one epoch on {device}: {time.time() - t:.2f}s ({len(train_dl)} batches of 128)")
    print("all model checks passed")
