# Vector Quantization

A PyTorch implementation of ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937), van den Oord et al, NIPS 2017. Unlike the official [sonnet repo](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py), this repo uses Discretized Logistic [1], as suggested by [Lucas Caccia](https://github.com/pclucas14/vq-vae)


Some of the VQ-VAE code is from [Zalando](https://github.com/zalandoresearch/pytorch-vq-vae), [Lucas Caccia](https://github.com/pclucas14/vq-vae), [Kim Seonghyeon](https://github.com/rosinality/vq-vae-2-pytorch), based on the official TensorFlow implementation from the [sonnet repo](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py).

References:
[1] Salimans, Tim, et al. "Pixelcnn++: Improving the pixelcnn with discretized
    logistic mixture likelihood and other modifications." ICLR 2017.
    https://arxiv.org/abs/1701.05517
