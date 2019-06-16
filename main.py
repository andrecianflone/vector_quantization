import argparse
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

def train_epoch(args, pbar, train_loader, model, optimizer,
        train_res_recon_error , train_res_perplexity):
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
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = torch.mean((data_recon - data)**2)/args.data_variance
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        # Print Average every 100 steps
        if (args.global_it+1) % 100 == 0:
            av_rec_err = np.mean(train_res_recon_error[-100:])
            av_ppl = np.mean(train_res_perplexity[-100:])
            pbar.print_train(av_rec_err=float(av_rec_err), av_ppl=float(av_ppl),
                                                                increment=100)
        args.global_it += 1

def evaluate(model, valid_loader):
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
    train_loader, valid_loader, data_var = data.get_data(args.data_folder,
                                                            args.batch_size)
    args.data_variance = data_var
    print(f"Training set size {len(train_loader.dataset)}")
    print(f"Validation set size {len(valid_loader.dataset)}")

    print("Loading model")
    model = VQVAE(args.num_hiddens, args.num_residual_layers,
                args.num_residual_hiddens, args.num_embeddings,
                args.embedding_dim, args.commitment_cost,
                args.decay).to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,
                                                                amsgrad=False)

    print(f"Start training for {args.num_epochs}")
    num_batches = math.ceil(len(train_loader.dataset)/train_loader.batch_size)
    pbar = Progress(num_batches, bar_length=20, custom_increment=True)
    best_loss = -1.

    train_res_recon_error = []
    train_res_perplexity = []
    args.global_it = 0
    for epoch in range(args.num_epochs):
        pbar.epoch_start()
        train_epoch(args, pbar, train_loader, model,
                optimizer, train_res_recon_error, train_res_perplexity)
        # loss, _ = test(valid_loader, model, args)
        # pbar.print_eval(loss)
        pbar.print_end_epoch()

    print("Plotting training results")
    utils.plot_results(train_res_recon_error, train_res_perplexity, "results/train.png")

    print("Evaluate and plot validation set")
    evaluate(model, valid_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Data settings
    add('--data_folder', type=str, default=".data/cifar10",
        help='Location of data (will download data if does not exist)')
    add('--dataset', type=str,
        help='Dataset name')
    add('--batch_size', type=int, default=32)
    add('--max_iterations', type=int, default=None,
            help="Use this for debugging (default: None)")
    add('--num_epochs', type=int, default=20,
        help='number of epochs (default: 20)')

    # Model, defaults like in paper
    add('--num_hiddens', type=int, default=128)
    add('--num_residual_hiddens', type=int, default = 32)
    add('--num_residual_layers', type=int, default=2)
    add('--embedding_dim', type=int, default=64,
            help='Embedding size, `D` in paper')
    add('--num_embeddings', type=int, default=512,
            help='Number of embeddings to choose from, `K` in paper')
    add('--commitment_cost', type=float, default=0.25,
            help='Beta in the loss function')
    add('--decay', type=float, default=0.99)
    add('--learning_rate', type=float, default=3e-4)

    args = parser.parse_args()
    args.device = device
    main(args)

