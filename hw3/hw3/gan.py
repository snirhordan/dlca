import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        number_of_channels = in_size[0]
        nn_sequential_input = []
        channels = [64,128,256,512]
        to_add = False
        for i in channels:
            block = [nn.Conv2d(number_of_channels, i, 4, stride=2, padding=1)]
            if not to_add:
                block +=[nn.LeakyReLU(0.2, inplace=True)]
            else:
                block +=[nn.BatchNorm2d(i), nn.LeakyReLU(0.2, inplace=True)]
            nn_sequential_input += block   
            number_of_channels = i
            to_add = True
        nn_sequential_input += [nn.Conv2d(number_of_channels, 1, 4, stride=1, padding=0)]
        self.dsc = nn.Sequential(*nn_sequential_input)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.dsc(x)
        y = y.squeeze(3)
        y = y.squeeze(2)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        cnn = [nn.ConvTranspose2d(z_dim, 512, 4, stride=1, padding=0)]
        channels = [512, 256, 128, 64]
        cnn += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        for in_channel, out_channel in zip(channels[:-1], channels[1:]):
            block = [nn.ConvTranspose2d(in_channel, out_channel, 4, stride=2, padding=1)]
            block +=[nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)]
            cnn += block
        cnn += [nn.ConvTranspose2d(channels[-1], out_channels, 4, stride=2, padding=1), nn.Tanh()]
        self.cnn = nn.Sequential(*cnn)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        torch.autograd.set_grad_enabled(with_grad)
        samples = self.forward(torch.randn([n, self.z_dim], device=device, requires_grad=with_grad))
        torch.autograd.set_grad_enabled(True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        x = self.cnn(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    function = nn.BCEWithLogitsLoss()
    added_const = -(label_noise / 2) + data_label
    noisy_label = torch.rand_like(y_data) * label_noise + added_const
    loss_data = function(y_data,noisy_label)
    added_const = added_const - 2*data_label + 1
    loss_generated = function(y_generated,torch.rand_like(y_generated) * label_noise + added_const)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    function = nn.BCEWithLogitsLoss()
    y = torch.zeros_like(y_generated) + data_label
    loss = function(y_generated, y)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_model.zero_grad()
    real_probabilty = dsc_model(x_data)
    new_probabilty = dsc_model(gen_model.sample(x_data.shape[0]).detach())
    dsc_loss = dsc_loss_fn(real_probabilty, new_probabilty)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_model.zero_grad()
    generated = gen_model.sample(x_data.shape[0],True)
    new_probabilty = dsc_model(generated)
    gen_loss = gen_loss_fn(new_probabilty)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    threshold = 0.1
    if len(dsc_losses) > 3:
        if abs(dsc_losses[-1] - dsc_losses[-2]) < threshold:
            if gen_losses[-1] < gen_losses[-2] and gen_losses[-2] < gen_losses[-3]:
                saved = True
                torch.save(gen_model, checkpoint_file)
    # ========================

    return saved
