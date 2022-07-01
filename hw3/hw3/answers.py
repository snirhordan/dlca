r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 500
    hypers['seq_len'] = 128
    hypers['h_dim'] = 500
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.01
    hypers['learn_rate'] = 0.0007
    hypers['lr_sched_factor'] = 0.08
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "When she was just a girl"
    temperature = 0.5
    # ========================
    return start_seq, temperature

part1_q1 = r"""
**Your answer:**
        
        since we are having an big corpus this means that loading it to memroy will take a long
        time and will make the processing slower, to avoid that we split that to small parts that we
        load every time.

"""

part1_q2 = r"""
**Your answer:**
            it possible that the generated text clearly shows memory longer than the sequence length because 
            we dont flush or reset the hidden state .
            Instead of that we tend to keep them and we pass it in time to the next seqeunce .
            The generated text learns the connections between the characters and have memory longer than 
            a single patch . 


"""

part1_q3 = r"""
**Your answer:**
	We are not shuffling the order of batches when training because we want to train the module with the right order
	of the batcher , keeping the correct order means that we are having a correct relation and order between the
	sentnces in the text which keeps it right logic between them , this means it would have in memry the right 
	context between the different parts of the text like they were in the original text.
	This should help our module to generate a text that is simmilar to the original text.


"""
part1_q4 = r"""
**Your answer:**


**a. We lower the temprature for the model to make the conditional distribution of the next word givn the current one as dissimilar to uniform distribution as possible. If the distribution were indeed uniform then taking maximum argument as criterion will yield very unpredictable and thus uninformative results.
**

**
b. Probability over the output with temparature T defined as $ e^{y_i/T} / \sum{e^{y_i/T}}  $
If T is very large than the exponent is very close to 0, then the numerator will be around 1 and denominator around n, then for any output we obtain a distribution similar to uniform distribution.
**

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=64,
        h_dim=512, z_dim=128, x_sigma2=0.0002,
        learn_rate=0.00004, betas=(0.9, 0.999)
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The hyperparameter $\sigma^2$ is used to set the distance between the encoding and the mean (describes the allowed difference between the distance and the mean.)
By using low sigma values, the images generated by the model are closer to the training data, that's because the model is closer to the mean and is more constrained by the data it has seen.
This is in contrast to using high sigma values, which may produce images that differ from the learned data.
"""

part2_q2 = r"""
**Your answer:**
1)Reconstruction Loss: Gives us a measure of how well the decoder reconstructs x. 
KL divergence loss: is a regularizer that measures how much information we lose when using q to represent p.

2)The effect of the KL loss on the latent-space distribution is as follows: the KL loss changes z_mu and z_sigma_2 given an instance of x by penalising the model to an inferior distribution of z.

3) The benefit of this effect lies in the improvement of the generation task, because it adds interpolations between classes and remove dicontinuities in the latent-space.
"""



part2_q3 = r"""
**Your answer:**
In the formulation of the VAE loss, we start by maximizing the evidence distribution, $p(\bb{X})$ because this helps us in finding the probability distrubuion of the data.
This means that maximizing $p(\bb{X})$ gives a propper aproximation of the actual distribuation of the data.
"""

part2_q4 = r"""
**Your answer:**
We use the log function here because we want to change the problem from a multiplication of all the the probabilities to a summation of the lof of all of those probabilities. We can use the log because it is monitinically ascending, and so the maximal value won't change.
 We can assume that because each data we got to train the model, is sampeld by the actual distibution."""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['z_dim'] = 100
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['betas'] =(0.5, 0.999)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
In the training phase, we only train the discriminator alone, and in this phase we sample the images accordingly. 
We don't want these samples to affect the gradient of the generator, so we need to separate these samples from the backpropagation process. 
This can happen even if we don't intend to. 
Therefore, when we train the generator and freeze the discriminator, we preserve these gradients to improve the sampling power of the generator.

"""

part3_q2 = r"""
**Your answer:**
1) We shouldn't decide to stop training just because the generator loss is below a certain threshold, because if we look at the results, we can see that a low loss rate doesn't mean that by definition, a GAN given a good image will have a loss here , here it is defined by the ability of the discriminator to detect fake images, it does not measure sample quality. Sometimes the discriminator is not very good and the generator produces bad samples, but these samples can fool the discriminator.

2) If the discriminator loss remains constant and the generator loss decreases, it means that the discriminator cannot correctly identify real and fake samples. Generator improved and created better samples.
"""

part3_q3 = r"""
**Your answer:**
It can be said that the images we generate with VAE are smoother and more focused on human faces. 
If we compare it to the VAE, those generated by the GAN are more noisy and have multiple colors. 
This might be due to the differences in architecture and loss function between both networks.
For example, if we compare the loss functions of these two: the VAE loss function is directly related to the dataset, unlike the GAN loss function, it is from a game theory perspective and has no direct relationship with the dataset, so the general picture related refers to the entire image, including the background and its colors. 
In the VAE dataset, we have a common face, and because of its architecture and care for mutual information in the input and decoded images, it preserves the common features in the resulting decoded images without preserving the background and its color .
"""

# ==============
