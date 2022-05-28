r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**<br />
**a. Output of layer is x*W^T+b. Shape of partial derivative is shape of W^t which is 512x1024** <br />
**b. W is sparse if it's initialized using a normal distribution with low std dev**<br />
**c. This is a linear transformation thus Jacobian of transformation w.r.t X is W^T. No need to explicitly calculate it.**<br />
Same process for derivative by W:<br />
**a. Shape of parital derivative of transformation w.r.t W is the shape of X which is 64x512**<br />
**b. X is sparse depending on the input. Not necessary for the model to learn efficiently, as opposed to W**
**c. The derivative w.r.t to W is X itself, as seen in implementatoin of linear layer, and therefore no need to calculate Jacobian explicitly.**
"""

part1_q2 = r"""
**Your answer:**

Backpropegation is not the only method of training neural networks. It is the most commonly used because in most classification tasks the output is a scalar and thus the gradients can be explicitly and  efficiently calculated.

In all models there must be some objective function that is minimized, not necessarily through backpropegation.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr = 3
    reg = 1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.05
    reg = 1.5
    lr_rmsprop = 0.5
    lr_vanilla = 0.01
    lr_momentum = 0.5
    # =======================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.05 
    lr = 1
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

** It is possible for both the cros entropy loss and the accuracy to simultanously increase during the testing phase.
Accuracy is the ratio of samples in the batch that are classified correcctly by taking the maximum argument in the probability vector inducd by cross entropy.
The loss is given by the equation $-y^t log(\hat{y}) $ which we have seen is equivalent to $ -x_i + log( e^{x_1} + ... +e^{x_n}) $ where i is the correct output label. 
The scenario in which the loss AND accuracy can both increase is if the entropy increases whilst the correct label retains the maximum probability.
Concretely, assume we have a mini vatch of two samples, although thhe cross entropy loss decreases in one it might increase more in the other thus the average cross entropy loss will increase, even though the classsification losss increases.

Take for example , in binary classification, in first epoch we get softmax activations : 
1. [0.1, 0.9]  , 2. [0.49 ,0.51 ] with correct labels 1. [0,1] 2. [1,0]
In second epoch we get : 
1. [ 0.4, 0.6 ] , 2. [0.51, 0.49] with correct labels 1.[0,1] 2. [1,0]

Accuracy of first mini-batch is 50% and for the second is 100%, so increased.

Loss of first mini-batch is $ [-0.9 + log(e^{0.1} + e^{0.9}) -0.49 + log( e^{0.49} + e^{0.51} ) ]/2 = 0.5371 $ 
loss of second mini-batch is $  [-0.6 + log(e^{0.4}+e^{0.6}) -0.51 + log( e^{0.49} + e^{0.51} ) ]/2 =0.64 $

Therefore both loss and accuracy increase.

Threfore for the first few epochs the classifier might get higher accuracy but the cross entropy loss  will increase, thereby yielding an increased loss. This issue in theory should be "fixed"  after a few batches when the crossentropy of what the model predicts and the true distribution decreases.
**

"""

part2_q3 = r"""
**Your answer:**

**a. Backpropegation is a method for training neural networks in which, using the chain rule, we are able to calculate the derivative of the loss function with respect to any of our parametrs ( weights, biases and such ).
Gradient descent is a general method of finding a local minimum of a function by taking "steps" along the direction of the gradient. From calculus we know the gradient points to the direction of steepest ascent, thereby negating the gradient goes in the direction of steepest descent. There is an implied assumption when training neural networks that if we update the parameters in the the direction of steepest descent of the value of the loss function w.r.t the parameters of the neural network we will reach a local, or as happens in practice usually, a global minimum of the loss function. 
**

**
b. Differences:

1. Computation resources : In gradient descent we calculate the gradient of the loss using the entire training dataset thus order N calculations in contrast with SGD where we sample uniformly ffrom the dataset thus have 1 calculation of the gradient of the loss function.
2. Guarantees on convergence : Using SGD we can't guarantee we are moving in the direction of steepest descent in EVERY iteration but only that the expected value over uniform distribution over the samples yields the gradient of th loss function. In gradient descent we calculate the gradient using N >> 1 samples then in this case the loss gradient closely resembles the expected value of the actual gradient of the loss function  in EVERY iteration 
3. Loss plot : Using SGD when far away from minima the convergence is more rapid than in standard gradient descent, yet when we are proximate to he minima there is a phemonema of staying at an asymptotically higher loss htan standard GD. That occurs because each update to the parameters s done according to a randomly picked sample thus the update direction never fully captures the optimal update direction close to the minima.
4. convergence rates : In SGD In k-th iteration distance between minima and k-iteration parametrs are sublinear, i.e. O(1/k), In Gradient descent the rate s linear and is O(c^k)
 ** <br \>
 **
 c. Observe computational resources in order to reach norm of less than $\epsilon >0$ in Euclidean norm, of the k-th iteration paramters and the minimzer, we need order of O(1/$\epsilon$) which is independent of N (training set size)
 when using GD the computational resources to reach same guarantee above our of order O(N log(1/$\epsilon$)) which is dependent on N. 
 
In real-world settings usually the dataset size is very large and computational resources are scarce thus SGD is more relavant.

Another factor is the quick minimization of loss of SGD in contrast with GD , even though asymptotically the loss is slightly higher using SGD than GD.
 
 ** <br \>
**
d. 1. The loss using GD is calculated as average loss for each sample.
      In mentioned approach, loss is average loss of batched summed up. 
** <br \>

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    activation = 'relu'
    hidden_dims = 10
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.001
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**1) Our train accuracy is relatively high until a certain threshold that limits its peak value. 
Our optimization error could be better by changing the models hyperparameters or optimizers. And thus changing the optimizer or the hyperparameters could yield a better optimization error.**

**2)There is a small difference between the loss of the test set and the loss of the training set. The reason behind this is that there is no overfitting to the training set and so we get a good generalization ability of our model  and a low generalization error.**

**3)When we talk about high approx. error we mainly refer to high loss.
Our training loss is relatively not low, which indicates a high approx. error.
One way ,through which we can decrease this error, is by creating a model with more channels.**
"""

part3_q2 = r"""
**When addressing the validation set, we expect to get FNR > FPR.
The reason for this assumption is due to the comparison between a positive number and a probability. That's why we expect a higher FNR on the validation set. 
This assumption was not totally correct, cause we can see from our runs that sometimes FPR can be higher or equal to FNR.**



"""

part3_q3 = r"""
**1) Our goal is to increase FPR relative to FNR. To achieve that we can increase the threshold in order to minimize the cost.
By minimizion the cost we decrease the FNR, which decreases the risk of not diagnosing the patient with non-lethal symptoms**

**2) In this case we want to increase the FNR relative to FPR. 
One we of achieving that can be through decreasing the threshold of diagnosing a patient with no clear lethat symptoms with a high probability of dying.**

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.2
    loss_fn = torch.nn.CrossEntropyLoss()
    momentum = 0.003
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**In our answer channel = feature map**

**1) If we ignore the bias then the number of parameters of a single convo. layer is:**

**(#channels_previous_layer * kernel_size^2)*#channels_current_layer **

**And so, we get:**

**For Regular Block:** Number of Parameters = (256 * 3^2) * 256 + (256 * 3^2) * 256 = 1179648

**For Bottelneck Block :** Number of Parameters = (256 * 1^2) * 64 + (64 * 3^2)*64 + (64 * 1^2) * 256 = 69632.

**2) The number of floating point operations in single Convo. layer is the number of parameters multiplied by 2*(dimenions_of_single_channel) **

**And so: **

**Bottleneck Block:** floating point operations = 2*32*32*((256 * 1^2) * 64 + (64 * 3^2) * 64 + (64 * 1^2) * 256) = 142,606,336

**Regular block:** floating point operations = 2*32*32*((256 * 3^2) * 256 + (256 * 3^2) * 256) = 2,415,919,104

** And so Our formula is 2*H*W *(k^2 * in * out), where k^2 is the multiplications to calculate the output of a single channel and (H*W) represents the feature map.**

**3) We see that the abitlity to combine the input spatially is better in the Regular Block.**

**The reason for that is that we apply 3 by 3 convolution that yields a receptive field that is wider than the one that we get from from the same method on Bottleneck Block.**

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
