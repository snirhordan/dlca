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
**b. X is sparse depending on the input. Not necessary for the model to learn efficiently, as opposed to W** <br />
**c. The derivative w.r.t to W is X itself, as seen in implementatoin of linear layer, and therefore no need to calculate Jacobian explicitly.**
"""

part1_q2 = r"""
**Your answer:**

Backpropegation is not the only method of training neural networks. It is the most commonly used because in most classification tasks the output is a scalar and thus the gradients can be explicitly and  efficiently calculated. Thus when using the chain rule we can easily calculate the infinitesimal change in the loss function w.r.t each of the parameters in the network.

In all models there must be some objective function that is minimized, not necessarily through backpropegation. For instance you can have an MLP 
where the last layer is trained using Least Squared optimizations, the previous layers are initialized randomly, and learning is done 
by pruning unnecessary connections within the previous to last layers. 
"""
# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.05, 0.05, 0
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
    wstd = 0.5
    reg = 0.0
    lr_rmsprop = 0.0002
    lr_vanilla = 0.01
    lr_momentum = 0.0042
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
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r""" 
**Your answer:**

The graphs match what we expected to see. <br/>
In high dropout setting, 0.8, we see decreased training accuracy compared with no dropout , because training a neural network with lower dropout has a higher propensity to overfit the data.<br />

Dropout is a regularization technique, thus we expect the NN to perform better on the testing dataset when regularization is done with high probability than with lower probability. <br />
In the test accuracy section, we have that the neural network with high dropout has higher accuracy because it generalizes better. <br />

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
**d
4. A. There is a difference in the two approches.
When training using GD during backpropegation we update the weights with multilplication of dout * w^t , with dout is of dimensions (d_out, N) and w is of dimensions (d_in, N).
In the approach described we calculate the change in w, as sum of matrices that are multiplications of (d_out, batch_size) by (batch_size, d_in).
Algebraically, these are distinct operations.

<br \>

B. We have to save in the cache the inputs for each linear layers for each one of the batches. Thus we have to save the same magnitude of memory as in GD but seuquentially. Therefore we will reach a point after saving the input for each layer of each one of the batcehs that the memory requiremnt will exceed our device's ability. That is because the device cannot handle the full GD thus in the process of incrementing the memory usage increasingly until reaching the same memory requirement as GD, we will run out of memory (mathemaatically the intermidate value theorem)

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
**1) From the results that we obtained, we can see that for a model with a depth of one, the decision boundary is very close to being linear. Once we played with the values of the model's width and kept depth=1 we managed to reach more complex decision boundaries.**

**We also noticed that when we increased the depth, the values of the width gained more importance. Once we increased the depth we also got more complex decision boundaries. The increase in depth also yielded models with better performance on the validation set.**


**2)Our results show  that when width is increasing we get a  better validation accuracy , and the threshold is being updated due to changing of the FPR and FNR.**

**3)For depth = 1 and width = 32 :  valid_acc = 93% and test_acc = 84.0%**

**For depth = 4 and width = 8 : valid_acc = 94% and test_acc = 90.1%**

**We got that the validation accuracy and test accuracy are both higher for the case of depth = 4 and width = 8. The reason for this is the fact that for depth = 1 the decision boundary is more linear and for depth = 4 it's more complex and, the roled of the width in this case is not noticeable.**

**For depth = 1 and width = 128 :  valid_acc = 92.9% and test_acc = 90.7%**

**For depth = 4 and width = 32 :  valid_acc = 93.7% and test_acc = 85.8%**

**Here we got that the decision boundary is more complex with a model with depth = 4 and thus the valid_acc is higher, but because the width is lower we get less test_acc**


**4)Validation set is used to fine tune the hyperparameters, also the value of the threshold is optimized using a validation set, and so, leads to better accuracy regarding the test_set.**

**And so, the hyperparameters that were chosen based on the validation set help us in generalizing our model and get the best test_set accuracy.**

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

1. Optimal results were when L = 4,8 when taking 64 filters and L=2,4 when using 32 layers.  It indicates a correct balance of both a lrage enough number of parametrs in the neural network and size that is not large enough to not be able to generalize. When the layer depth is too small the neural network isn't expressive enough, i.e. the function space it spans isn't rich enough to represent the udnerlying distribution of the dataset.

2. When the network depth is too large the model becomes untrainable due to the vanishing gradients phenomenon. This issue can be solved using residual blocks, which pass the convolution layers and thereby avoiding the vanishing gradients problem. Additionally, when trainin deep neural networks the distribution on the ouputs of each layer changes with different batches resulting in slower convergence. When normalizing the batch the distribution over the layers' inputs is regularized. 


"""

part5_q2 = r"""

This experiment compares the effectiveness of the number of convolutional filters with varying depths of networks. 
When the depth of the neural network increases, so does increasing the number of filters effects positively on the performence of the network.
Concretely, best testing accuracies where (L=2, K=32), (L=4, K=64), (L=8, k=256)

All the models were trainable, in contrast to exp 1.1. In experiment 1.1 we isolated the number of layers as the parameter we are changing and n the second experiment we are interested in observing the change in training isolating the number of features.

"""

part5_q3 = r"""

On this experiment we observe that all neural networks perform relatively similarly on the test accuracy, with the depth 4 taking most time to converge. This is expected because there are more parameters to tune.

The training accuracy of the shallow networks is much higher than the test accuracy which means the overfit the data. The deeper neural network has most similar training/test accuracies, which means they generalize best.

"""

part5_q4 = r"""

We see in experiment 4 in the first part when k = [32], when the depth is too large the model trains slower and isn't able to generalize as well as those with lower depths (depth 8,16)

Yet when K=[64, 128, 256], we get the even the model with most depth is able to generalize well. This is hown in the similarly high test accuracies for models of all depths in this case.  A reason for this is the choice of ascending number of features which cature "finer" details of the images.

Comparing exp 1.4 to exp 1.1 and 1.3, we see that we get better test accuracies. This implies that using increasing number of feature values in the architecture is beneficial for learning images.


"""

part5_q5 = r"""

1. Main differences was using Residual Blocks, which prevent gradient vanishing, dropout, and batchnorm. These arose from conclusions from the previous experiments. 

2. Model test accuracy much increased from previous experiments when using 12 layers in experiment 2. 
Compared to experiment 1, the model generalizes much betterand  all models configurations were trainable.
"""
# ==============
