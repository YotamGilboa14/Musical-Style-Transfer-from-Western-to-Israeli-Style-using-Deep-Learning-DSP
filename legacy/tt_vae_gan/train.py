"""Legacy VAE-GAN proof-of-concept - training script (superseded).

Part of the early trumpet-to-violin timbre-transfer experiment we built before
switching to the diffusion model. Kept for reference and learning only; the
final pipeline does not use it. See legacy/README for the full context.
"""

# Forked from ebadawy/voice_conversion/src/train.py
# CycleGAN-style VAE-GAN training for timbre transfer

import argparse
import os
import numpy as np
import itertools
import random
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

# Use relative imports when run as module, absolute when run as script
try:
    from .models import (Encoder, Generator, Discriminator, ResidualBlock,
                         weights_init_normal, LambdaLR)
    from .data_proc import DataProc
    from .utils import plot_batch_train
    from .params_config import get_params as _get_params
except ImportError:
    from models import (Encoder, Generator, Discriminator, ResidualBlock,
                        weights_init_normal, LambdaLR)
    from data_proc import DataProc
    from utils import plot_batch_train
    from params_config import get_params as _get_params


def compute_kl(mu):
    """KL divergence loss for the latent space (vs standard normal)."""
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


def train(opt):
    """
    Main training loop for the VAE-GAN timbre transfer model.
    
    Args:
        opt: argparse namespace with training configuration
    """
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Create checkpoint directory
    os.makedirs("saved_models/%s" % opt.model_name, exist_ok=True)

    # ─── Losses ─────────────────────────────────────────────────────────────
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Dimensionality (channel-wise) of image embedding
    shared_dim = opt.dim * 2 ** opt.n_downsample

    # ─── Initialize models ──────────────────────────────────────────────────
    encoder = Encoder(dim=opt.dim, in_channels=opt.channels,
                      n_downsample=opt.n_downsample)
    shared_G = ResidualBlock(features=shared_dim)
    G = [Generator(dim=opt.dim, out_channels=opt.channels,
                   n_upsample=opt.n_downsample, shared_block=shared_G)
         for _ in range(opt.n_spkrs)]
    D = [Discriminator(input_shape) for _ in range(opt.n_spkrs)]

    if cuda:
        encoder = encoder.cuda()
        G = [G_n.cuda() for G_n in G]
        D = [D_n.cuda() for D_n in D]
        criterion_GAN.cuda()
        criterion_pixel.cuda()

    # ─── Load or initialize weights ─────────────────────────────────────────
    if opt.epoch != 0:
        encoder.load_state_dict(torch.load(
            "saved_models/%s/encoder_%02d.pth" % (opt.model_name, opt.epoch - 1)))
        for n in range(opt.n_spkrs):
            G[n].load_state_dict(torch.load(
                "saved_models/%s/G%d_%02d.pth" % (opt.model_name, n + 1, opt.epoch - 1)))
            D[n].load_state_dict(torch.load(
                "saved_models/%s/D%d_%02d.pth" % (opt.model_name, n + 1, opt.epoch - 1)))
    else:
        encoder.apply(weights_init_normal)
        for n in range(opt.n_spkrs):
            G[n].apply(weights_init_normal)
            D[n].apply(weights_init_normal)

    # ─── Loss weights ───────────────────────────────────────────────────────
    lambda_0 = 10    # GAN
    lambda_1 = 0.1   # KL (encoded spect)
    lambda_2 = 100   # ID pixel-wise
    lambda_3 = 0.1   # KL (encoded translated spect)
    lambda_4 = 100   # Cycle pixel-wise
    lambda_5 = 10    # Latent space L1

    # ─── Optimizers ──────────────────────────────────────────────────────────
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(),
                        *[G_n.parameters() for G_n in G]),
        lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = [
        torch.optim.Adam(D_n.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        for D_n in D]

    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D = [
        torch.optim.lr_scheduler.LambdaLR(
            opt_D_n,
            lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        for opt_D_n in optimizer_D]

    # ─── Dataloader ──────────────────────────────────────────────────────────
    dataloader = DataLoader(
        DataProc(opt, split='train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True)

    # ─── Training loop ──────────────────────────────────────────────────────
    for epoch in range(opt.epoch + 1, opt.n_epochs + opt.epoch):
        losses = {'G': [], 'D': []}
        progress = tqdm(enumerate(dataloader), desc='', total=len(dataloader))

        for i, batch in progress:
            # For each target, randomly choose a source for training
            for trg_id in range(opt.n_spkrs):
                potential_src_ids = list(range(opt.n_spkrs))
                potential_src_ids.pop(trg_id)
                src_id = random.choice(potential_src_ids)

                # ─── Train Encoder + Generators ──────────────────────────
                # Create plot output directories
                if opt.plot_interval != -1:
                    os.makedirs("out_train/%s/plot_%dt%d/" % (
                        opt.model_name, src_id, trg_id), exist_ok=True)

                X1 = Variable(batch[src_id].type(Tensor))
                X2 = Variable(batch[trg_id].type(Tensor))

                valid = Variable(Tensor(np.ones((X1.size(0),
                    *D[trg_id].output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((X1.size(0),
                    *D[trg_id].output_shape))), requires_grad=False)

                optimizer_G.zero_grad()

                # Encode
                mu1, Z1 = encoder(X1)
                mu2, Z2 = encoder(X2)

                # Latent features
                feat_1 = mu1.view(mu1.size()[0], -1).mean(dim=0)
                feat_2 = mu2.view(mu2.size()[0], -1).mean(dim=0)

                # Reconstruct
                recon_X1 = G[src_id](Z1)
                recon_X2 = G[trg_id](Z2)

                # Translate
                fake_X1 = G[src_id](Z2)
                fake_X2 = G[trg_id](Z1)

                # Cycle translate
                mu1_, Z1_ = encoder(fake_X1)
                mu2_, Z2_ = encoder(fake_X2)
                cycle_X1 = G[src_id](Z2_)
                cycle_X2 = G[trg_id](Z1_)

                # Compute losses
                loss_GAN_1 = lambda_0 * criterion_GAN(D[src_id](fake_X1), valid)
                loss_GAN_2 = lambda_0 * criterion_GAN(D[trg_id](fake_X2), valid)
                loss_KL_1 = lambda_1 * compute_kl(mu1)
                loss_KL_2 = lambda_1 * compute_kl(mu2)
                loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
                loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
                loss_KL_1_ = lambda_3 * compute_kl(mu1_)
                loss_KL_2_ = lambda_3 * compute_kl(mu2_)
                loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
                loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)
                loss_feat = lambda_5 * criterion_pixel(feat_1, feat_2)

                loss_G = (loss_KL_1 + loss_KL_2 + loss_ID_1 + loss_ID_2
                          + loss_GAN_1 + loss_GAN_2 + loss_KL_1_ + loss_KL_2_
                          + loss_cyc_1 + loss_cyc_2 + loss_feat)

                loss_G.backward()
                optimizer_G.step()

                # ─── Train Discriminators ────────────────────────────────
                optimizer_D[src_id].zero_grad()
                loss_D1 = (criterion_GAN(D[src_id](X1), valid)
                           + criterion_GAN(D[src_id](fake_X1.detach()), fake))
                loss_D1.backward()
                optimizer_D[src_id].step()

                optimizer_D[trg_id].zero_grad()
                loss_D2 = (criterion_GAN(D[trg_id](X2), valid)
                           + criterion_GAN(D[trg_id](fake_X2.detach()), fake))
                loss_D2.backward()
                optimizer_D[trg_id].step()

                # ─── Plot training progress ──────────────────────────────
                if (opt.plot_interval != -1
                        and (epoch + 1) % opt.plot_interval == 0
                        and i == 0):
                    plot_batch_train(opt.model_name,
                                    'plot_%dt%d' % (src_id, trg_id),
                                    epoch, X1, cycle_X1, fake_X2, X2)

                losses['G'].append(loss_G.item())
                losses['D'].append((loss_D1 + loss_D2).item())

            # Update progress bar
            progress.set_description(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (
                    epoch, opt.n_epochs + opt.epoch,
                    np.mean(losses['D']), np.mean(losses['G'])))

            # Update learning rates
            lr_scheduler_G.step()
            for n in range(opt.n_spkrs):
                lr_scheduler_D[n].step()

        # ─── Save checkpoints ───────────────────────────────────────────
        if (opt.checkpoint_interval != -1
                and epoch % opt.checkpoint_interval == 0):
            torch.save(encoder.state_dict(),
                       "saved_models/%s/encoder_%02d.pth" % (opt.model_name, epoch))
            for n in range(opt.n_spkrs):
                torch.save(G[n].state_dict(),
                           "saved_models/%s/G%d_%02d.pth" % (opt.model_name, n + 1, epoch))
                torch.save(D[n].state_dict(),
                           "saved_models/%s/D%d_%02d.pth" % (opt.model_name, n + 1, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0,
                        help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")
    parser.add_argument("--model_name", type=str, required=True,
                        help="name of the model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="path to dataset for training")
    parser.add_argument("--n_spkrs", type=int, default=2,
                        help="number of speakers/instruments")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of second order momentum")
    parser.add_argument("--decay_epoch", type=int, default=50,
                        help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads for batch generation")
    parser.add_argument("--img_height", type=int, default=128,
                        help="size of image height (mel bins)")
    parser.add_argument("--img_width", type=int, default=128,
                        help="size of image width (frames)")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--plot_interval", type=int, default=-1,
                        help="epoch interval for saving plots (-1 to disable)")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="interval between saving model checkpoints")
    parser.add_argument("--n_downsample", type=int, default=2,
                        help="number downsampling layers in encoder")
    parser.add_argument("--dim", type=int, default=32,
                        help="number of filters in first encoder layer")
    parser.add_argument("--use_pipeline_params", action="store_true",
                        help="use pipeline params (80 mels, 22050 Hz) "
                             "instead of original (128 mels, 16000 Hz)")

    opt = parser.parse_args()

    # Activate pipeline params BEFORE any param-dependent imports are used
    if opt.use_pipeline_params:
        os.environ['TT_VAE_GAN_USE_PIPELINE'] = '1'
        # Re-initialize params config to pick up the env var
        try:
            from .params_config import use_pipeline_params
            p = use_pipeline_params()
        except ImportError:
            from params_config import use_pipeline_params
            p = use_pipeline_params()
        # Override img_height to match pipeline mels
        opt.img_height = p.num_mels
        print(f"Using PIPELINE params: {p.num_mels} mels, {p.sample_rate} Hz")
    else:
        print(f"Using ORIGINAL params: 128 mels, 16000 Hz")

    print(opt)

    train(opt)
