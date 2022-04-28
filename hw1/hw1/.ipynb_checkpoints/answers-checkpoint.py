r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

        1.False : The test set gives us an indication regarding the generalization ability of our model.
                  The in-sample error on the other hand is the error rate that we get using our predictor on the same dataset we used to train our predictor,
                  i.e. in our case the training dataset.
                  The test set is comprised of samples the predictor did not train on and therfore does not affect the in-sample error.
                  
        2.False : Not every ratio of splitting the data into train and test set is useful.
                  We use the train set in order to train,build and fine tune our model. Subsequently, we check the generalization ability of that model using 
                  the test set.
                  It is conventional to split the entire dataset into 80% train-set and 20% test set.
                  If the train set is very small, then we will get a larSge approximation error. The model will not estimate a reasonable function underlying 
                  our data.
                  If the train set is too large, then this might lead to overfitting the model to the train set that we are building, which lead to a poor
                  generalization ability of our model.
                  
        3.True : Using the test set is done in order to check the generalization ability of the model. 
                 In case it is used in the cross validation then it would change the models hyperparameters 
                 and lead to a model with poor generalization ability, so it should not be used in cross validation.
        
        4.False : After performing K-fold cross validation, we average the validation-set performence over all the folds and use that average as the proxy 
                  for the model's generalization error.
                  


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**
    Although regularization done right would improve the model's generalization, in this case, it is not performed correctly. Choosing the hyperparameters of 
    the model based on perfoemence against the test dataset will lead to overfitting, i.e. a high generalizaiton error.
    The optimal is using k-fold validation to choose the right hyperparameters without accessing the test dataset.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

        Increasing k to a certain amount would indeed help us develop a more generalized model.
        On the other hand, if the value of k becomes too great, then the classification would 
        be decided by the most dominant class in the dataset (more dominant label type).
        So for every new sample, the decision would be based on the dominant type rather than on the 
        sample's characteristics.
        A too small value of k would yield a model that is affected by noisy samples, thus the variance would be higher.
        Thus we should pick a k that complies with a certain tradeoff and deliver good results.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

    1. This approach delivers models that are very much dependant on the data analyzed in the train set,
       thus it may lead to a model which overfits itself to the data that it knows already, and so the generalization
       ability of this model is poor.
       On the other hand, when using K-fold CV, during each fold we train the model on a subset of the training dataset
       and validate it against the rest of the  training dataset. In this way, we don't train and validate the model on the same data (in each fold).
    2. This approach yields a model that overfits the test dataset. That is because when choosing a model that minimizes the test error, if the test dataset is 
        too small or not representative of the underlying distribution of the entire dataset, thus the model we picked would have have high generalization error.
    
        When using k-fold cross validation, when picking the best model, for example using different hyperparameters of some model, we evaluate each model's 
    average performence over all the folds. 
    
    Each training-validation fold is independent of the test dataset. 
    This has a twofold advantage: during every fold we don't access the test dataset and therefore don't overfit on the ultimate criterion of the model and
    averaging the performence over several folds, rather than relying on one as in the train-test scenario, diminishes overfitting on the validation dataset.  
       

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
    Delta should be positive in order to assure that the gradient converges towards the class of the 
    highest score aka. towards the class which gives us minimal errors.
    The value is chosen in an arbitrary way for the SVM loss  𝐿(𝑾) because The hyperparameters Δ and λ seem like two different hyperparameters,
    but in fact they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective.
    The key to understanding this is that the magnitude of the weights W has direct effect on the scores (and hence also their differences):
    As we shrink all values inside W the score differences will become lower, and as we scale up the weights the score differences will all become higher.
    

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**
    
    1.The linear model produces an avarage representation per class , when the model sees a familiar image, it classifies it 
     according to the most similar representation.
     Essentially the classifier learns which areas of the image is most likely to be "activated" for each digit. In the weight images this can be seen as particular regions of the canvas having more weight than others.
     For example, the digit zero has weights of high magnitude spread out over a circular shape, in contrast with the weights of the digit 3, which schematically have lower valued weughts where one would close it to make a circle. That differentiates the digit 3 from 0.
     Errors occur when an input digit is in a shape which resembles two distinct weight matrices. For example in the first row of printouts, the digit 5 is drawn with the bottom alf enclosed in a circle. In the weight activation for the digit 5, there is low weights for the region where the bottom half of 5 encloses into a ball, whereas the weight matrix for the digit 6 has high weights, for an enclosed loop. This is why the resulting value of the weight matrix that corresponds to 6 with the image had a higher value than the multiplication of the weight matrix corresponding to the digit 5 with the input image. 
     As a result of the above issue, we noticed some falsely classified images.

    2. K-nn is a naive algorithm, it classifies the sample according to its K nearest neighbours in its dataset.
       The linear model learns features over the entire dataset and given a new instance classifies it according to which regions it is mostly written in and which digit is most likely to be written in those regions based on the entire training dataset.
       k-NN only classifies in accordance with the similarity of the new instance with its k neighbors, which don't even snecessarily contain all the available labels. 
       This means the linear model generalizes the regions where each digit is written by learning a representation of likely regions activated through training over the entire training dataset. In contrast, the k-NN model only takes into account local behavior of the dataset and naive similarity.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

        1. If the learning rate is good, then we would smoothly converge to zero.
           If it was too high, then we would get a noisy graph, that might not converge at all.
           If it wass too slow, the graph would behave similarily to the behaviour of a good learning rate
           but it won't reach a point where the loss is low, beccause it would take us too much time to get there.

        2.The model is slightly overfitted to the training set, we concluded this because of the slightly higher accuracy in when
        addressing the performance of the model on the training set, the accuracy on the validation set is only slightly lower than
        that on the training set.
        So we concluded that our model only slightly overfits the training set.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
    The ideal pattern to see in the residual plot consists of points located exactly on the line.
    We should get minimal difference (distance) between the predictions and the labels.
    In our model we got dense points around the horizontal axis, which indicates a good fitting for the model.
    
    We got a final plot after CV which is much better than the one with the top 5 features, the average distances between the 
    samples and the lines are smaller, which means better performance in terms of calssification.
    
    Additionally the outliers classified by the final model after CV have much lower distance from the horizontal line than then the outliers from the model relying on just the top 5 features. Outliers is an ambiguous term but we can approximately define them as those that have a difference of more then 15 from the horizontal line. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**
    In general adding non-linear transformations to our dataset lets the classifier train on a more descriptive higher-dimensional space where a linear 
    regresson might better fit the dataset.
    
    1. We can look at the model as a linear regression with higher dimensions, we can see that the non-linearity in our case is not
       in the model and its parameters, rather in the dataset itself.
        
    2. When fittimg non-linear transformation on the dataset we expect the dataset to "behave" more like a linear function on the higher dimensional space and 
    thus giving us a better classification result using Linear Regression on that space.
    The only impediment to transformin the data is if the non-linear function is not defined for some valuse of a feature. Otherwise, there is no restriction 
    on what non-linearity to use.
    
    
    3. The effect of adding non-linear features to our data is the same as adding more features to our data. 
       The model can generalize relationships among the features that are more intricate.
       In Section 3, we fed the model untempered data and recieved a separating hyperplane.
       When we add more features, we undergo the same training as before, i.e. fimdimg the linear classifier for each class with best margin, yet on a higher 
       dimensional feature space. 
       Our resulting classfier will remain linear, that is the decision boundary will be a hyperplane in some higher dimensional feature space, depenedent on 
       the feature transformations we applied.
       

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**

    1. 
    In the logspace case we get that the effect between jumps of consecutive numbers is negligible, 
      however a change in jumps of 10 times can be seen.
      In other words, this approach focuses on significant changes.
      When using CV to choose best hyprparameters we want to minimze the number of combination of hyperparamers we check. Checking too many hyperparametrs that only pose a negligible improvement in the model wastes memory and computation resources. 
      In this case, the values of $\lambda$ affect the performence of the model similarly when are in some range of numbers, therefore there is no need to check each number in the range but to sample one of them. Using lospace we iterate over an exponentially increasing sequence in which each element increases drastically from the previous one.
      
     2. The model was fitted 180 times.
        This can be explained because we have 60 different combinations of the parameters, and 
        for each of these combintations we do 3-fold CV.
        So in total 180 times.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
