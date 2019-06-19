"""
Vector Quantization

References:
[1] Salimans, Tim, et al. "Pixelcnn++: Improving the pixelcnn with discretized
    logistic mixture likelihood and other modifications." ICLR 2017.
    https://arxiv.org/abs/1701.05517

[2] van den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation
    learning." Advances in Neural Information Processing Systems. 2017.
    https://arxiv.org/abs/1711.00937

"""
import argparse
import os
import math
import data
import utils
import torch
import torch.optim as optim
from vqvae import VQVAE
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from progress import Progress

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(args, pbar, valid_loader, model):
    """
    Evaluate validation set
    """
    model.eval()
    valid_loss, valid_recon_error , valid_perplexity = [], [], []
    # Loop data in validation set
    for data, _ in valid_loader:

        data = data.to(args.device)

        # Loss
        vq_loss, data_recon, perplexity = model(data)
        recon_error = torch.mean((data_recon - data)**2)/args.data_variance
        loss = recon_error + vq_loss

        valid_loss.append(loss.item())
        valid_recon_error.append(recon_error.item())
        valid_perplexity.append(perplexity.item())

    av_loss = np.mean(valid_loss)
    av_rec_err = np.mean(valid_recon_error)
    av_ppl = np.mean(valid_perplexity)
    pbar.print_eval(av_loss)
    # pbar.print_train(av_rec_err=float(av_rec_err), av_ppl=float(av_ppl),
                                                        # increment=100)
    return av_loss

def train_epoch(args, pbar, train_loader, model, optimizer,
        train_bpd, train_res_recon_error , train_res_perplexity, KL, N):
    """
    Train for one epoch
    """
    model.train()
    # Loop data in epoch
    for data, _ in train_loader:

        # This break used for debugging
        if args.max_iterations is not None:
            if args.global_it > args.max_iterations:
                break

        data = data.to(args.device)

        # Get reconstruction and vector quantization loss
        # `x`: reconstruction of `input`
        # `vq_loss`: MSE(encoded embeddings, nearest emb in codebooks)
        # x_prime, vq_loss, perplexity = model(data)
        x_prime, vq_loss = model(data)

        # Use Discretized Logistic as an alternative to MSE, see [1]
        log_pxz = utils.discretized_logistic(x, model.dec_log_stdv,
                                                        sample=input).mean()

        # recon_error = torch.mean((data_recon - data)**2)/args.data_variance
        # loss = recon_error + vq_loss

        loss = -1 * (log_pxz / N) + args.commit_coef * latent_loss

        elbo = - (KL - log_pxz) / N
        bpd  = elbo / np.log(2.)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bpd.append(bpd.item())
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        # Print Average every 100 steps
        if (args.global_it+1) % 100 == 0:
            av_rec_err = np.mean(train_res_recon_error[-100:])
            av_ppl = np.mean(train_res_perplexity[-100:])
            pbar.print_train(av_rec_err=float(av_rec_err), av_ppl=float(av_ppl),
                                                                increment=100)
        args.global_it += 1

def generate_samples(model, valid_loader):
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    (valid_originals, _) = next(iter(valid_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    utils.show(valid_reconstructions, "results/valid_recon.png")
    utils.show(valid_originals, "results/valid_originals.png")

def main(args):
    print("Loading data")
    train_loader, valid_loader, data_var, input_size = \
                                data.get_data(args.data_folder,args.batch_size)

    args.input_size = input_size
    args.downsample = args.input_size[-1] // args.args.enc_height
    args.data_variance = data_var
    print(f"Training set size {len(train_loader.dataset)}")
    print(f"Validation set size {len(valid_loader.dataset)}")

    print("Loading model")
    model = VQVAE(args).to(device) # see [2]
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,
                                                                amsgrad=False)

    print(f"Start training for {args.num_epochs} epochs")
    num_batches = math.ceil(len(train_loader.dataset)/train_loader.batch_size)
    pbar = Progress(num_batches, bar_length=20, custom_increment=True)

    # Needed for bpd
    # TODO: HERE
    KL = args.enc_height * args.enc_height * args.num_codebooks * \
                                                    np.log(args.num_embeddings)
    N  = np.prod(args.input_size)

    best_valid_loss = float('inf')
    train_bpd = []
    train_res_recon_error = []
    train_res_perplexity = []
    args.global_it = 0
    for epoch in range(args.num_epochs):
        pbar.epoch_start()
        train_epoch(args, pbar, train_loader, model, optimizer, train_bpd,
                            train_res_recon_error, train_res_perplexity, KL, N)
        # loss, _ = test(valid_loader, model, args)
        # pbar.print_eval(loss)
        valid_loss = evaluate(args, pbar, valid_loader, model)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
        pbar.print_end_epoch()

    print("Plotting training results")
    utils.plot_results(train_res_recon_error, train_res_perplexity,
                                                        "results/train.png")

    print("Evaluate and plot validation set")
    generate_samples(model, valid_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Data and training settings
    add('--data_folder', type=str, default=".data/cifar10",
            help='Location of data (will download data if does not exist)')
    add('--dataset', type=str,
            help='Dataset name')
    add('--batch_size', type=int, default=32)
    add('--max_iterations', type=int, default=None,
            help="Max it per epoch, for debugging (default: None)")
    add('--num_epochs', type=int, default=20,
            help='number of epochs (default: 20)')
    add('--learning_rate', type=float, default=3e-4)

    # Quantization settings
    add('--num_codebooks', type=int, default=1,
            help='Number of codebooks')
    add('--embed_dim', type=int, default=64,
            help='Embedding size, `D` in paper')
    add('--num_embeddings', type=int, default=512,
            help='Number of embeddings to choose from, `K` in paper')
    add('--commitment_cost', type=float, default=0.25,
            help='Beta in the loss function')
    add('--decay', type=float, default=0.99,
            help='Moving av decay for codebook update')

    # Model, defaults like in paper
    add('--enc_height', type=int, default=8,
            help="Encoder output size, used for downsampling and KL")
    add('--num_hiddens', type=int, default=128,
            help="Number of channels for Convolutions, not ResNet")
    add('--num_residual_hiddens', type=int, default = 32,
            help="Number of channels for ResNet")
    add('--num_residual_layers', type=int, default=2)

    # Misc
    add('--saved_model_name', type=str, default='vqvae.pt')
    add('--saved_model_dir', type=str, default='saved_models/')
    add('--seed', type=int, default=521)

    args = parser.parse_args()

    # Extra args
    args.device = device
    args.save_path = os.path.join(args.saved_model_dir, args.saved_model_name)
    utils.maybe_create_dir(args.saved_model_dir)

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)

