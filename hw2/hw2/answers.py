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
b. Gradient decent is a 

**
Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
