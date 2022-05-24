import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        self.conv_out_w = in_w
        self.conv_out_h = in_h
        counter = 0
        channelsIT = zip ([in_channels] + self.channels[:-1], self.channels[:])
        pooling_function = POOLINGS[self.pooling_type]
        activation_function = ACTIVATIONS[self.activation_type](**self.activation_params)
        for in_channel,out_channel in channelsIT :
            layers += [nn.Conv2d(in_channel, out_channel, **self.conv_params)]
            layers += [activationFunction]
            numerator = self.conv_out_w 
            denominator = self.conv_params["stride"]
            added_value = - (self.conv_params["kernel_size"]) + 2 * self.conv_params["padding"]
            numerator = numerator + added_value
            self.conv_out_w = math.floor(numerator / denominator)
            numerator = self.conv_out_h
            numerator = numerator + added_value
            self.conv_out_h = math.floor(numerator / denominator)
            counter += 1
            modulo_result = counter % self.pool_every
            if modulo_result == 0:
                layers += [poolingFunction(**self.pooling_params)]
                tmp = self.pooling_params["kernel_size"]
                self.conv_out_w = math.floor(self.conv_out_w / tmp)
                self.conv_out_h = math.floor(self.conv_out_h / tmp)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            convolution = torch.unsqueeze(torch.zeros(self.in_size),0)
            n_features = torch.numel(self.feature_extractor(convolution))
            return n_features
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        activation_function = ACTIVATIONS[self.activation_type](**self.activation_params)
        mlp =MLP(in_dim= self._n_features(), dims= self.hidden_dims+[self.out_classes], nonlins=[*[activation_function]*(len(self.hidden_dims)), 'none'])
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = (self.feature_extractor(x)).reshape(out.shape[0], -1)
        out = self.mlp(out)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_layers = []
        channel_aggregate = [in_channels] + channels[:]
        for i in range (len(kernel_sizes)) :
            mainLayers += [nn.Conv2d(channel_aggregate[i],channel_aggregate[i+1],kernel_size=kernel_sizes[i],bias=True, padding='same')]
            if dropout > 0:
                main_layers += [nn.Dropout2d(dropout)]
            if batchnorm is True:
                main_layers += [nn.BatchNorm2d(channel_aggregate[i+1])]
            main_layers += [ACTIVATIONS[activation_type](**activation_params)]
        self.main_path = nn.Sequential(*main_layers)
        shortcut_layers = [nn.Identity()]
        if in_channels != channels[-1] :
            shortcut_layers += [nn.Conv2d(in_channels,channels[-1], kernel_size=1, bias=False, padding='same')] 
        self.shortcut_path =  nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        super().__init__(**kwargs,in_channels = in_out_channels,channels = ([inner_channels[0]]+ inner_channels + [in_out_channels]),kernel_sizes = ([1]+inner_kernel_sizes+[1]))
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        # ====== YOUR CODE: ======
        aggregated_channels = [in_channels] + self.channels
        pools_num = len(self.channels) // self.pool_every
        for i in range (pools_num):
            if aggregated_channels[i*self.pool_every] != aggregated_channels[self.pool_every*(i+1)] or self.bottleneck == False: 
                layers += [ResidualBlock(aggregated_channels[i * self.pool_every], aggregated_channels[i * self.pool_every + 1: (i + 1) * self.pool_every + 1], kernel_sizes=[3] * self.pool_every, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params)]
            else:
                in_out_channels = int(aggregated_channels[i * self.pool_every])
                inner_channels= aggregated_channels[i * self.pool_every+2: (i+1)*self.pool_every]
                layers += [ResidualBottleneckBlock(in_out_channels=in_out_channels, inner_channels=inner_channels, inner_kernel_sizes = [3] * (self.pool_every-2),batchnorm=self.batchnorm,  dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params)]     
            layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
        remain = len(self.channels) % self.pool_every
        if remain > 0:
            if aggregated_channels[-remain - 1] != aggregated_channels[-1] or self.bottleneck == False:
                layers += [ResidualBlock(aggregated_channels[-remain - 1], aggregated_channels[-remain:],kernel_sizes=[3] * remain, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,activation_params=self.activation_params)]
            else:
                in_out_channels = int(aggregated_channels[-remain - 1])
                inner_channels= aggregated_channels[-remain:-1]
                layers += [ResidualBottleneckBlock(in_out_channels=in_out_channels, inner_channels=inner_channels,inner_kernel_sizes = [3] * (p-2), batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params)]
        self.conv_out_w = in_w // (self.pooling_params["kernel_size"] ** pools_num)
        self.conv_out_h = in_h // (self.pooling_params["kernel_size"] ** pools_num)
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCNN(CNN):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims, batchnorm=True, activation_type='lrelu', activation_params=dict(negative_slope=0.01),pooling_type='max', pooling_params=dict(kernel_size=2), dropout=0.1, **kwargs):
        """
        See CNN.__init__
        """
        
        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims, activation_type=activation_type, activation_params=activation_params, pooling_type=pooling_type, pooling_params=pooling_params)
        self.dropout = dropout
        self.batchnorm = batchnorm
        # ========================

    # TODO: Change whatever you want about the CNN to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        layers = []
        in_channels, in_h, in_w, = tuple(self.in_size)
        all_channels = [in_channels] + self.channels
        pools_num = len(self.channels) // self.pool_every
        for i in range (pools_num):
            layers += [ResidualBlock(all_channels[i * self.pool_every], all_channels[i * self.pool_every + 1: (i + 1) * self.pool_every + 1],kernel_sizes=[3] * self.pool_every, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params)]
            layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
        remain = len(self.channels) % self.pool_every
        if remain > 0: 
            layers += [ResidualBlock(all_channels[-remain - 1], all_channels[-remain:],kernel_sizes=[3] * remain, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, activation_params=self.activation_params)]
        self.conv_out_w = in_w // (self.pooling_params["kernel_size"] ** pools_num)
        self.conv_out_h = in_h // (self.pooling_params["kernel_size"] ** pools_num)    
        sequential = nn.Sequential(*layers)
        return sequential
    # ========================
