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
                  The in-sample error on the other hand is the error rate that we got in the data set we used for building our model.
                  That is why the test set does not affect the in-sample error because there it's only tested after training the model.
                  
        2.False : Not every ratio of splitting the data into train and test set is useful.
                  We use the train set in order to train,build and fine tune our model and check the generalization ability of that model using the test set.
                  It is conventional to split the entire dataset into 80% train-set and 20% test set.
                  If the train set is very small, then it will badly affect the model that we are building.
                  If the train set is too large, then this might lead to overfitting the model to the train set that we are building, which lead to a poor
                  generalization ability of our model.
                  
        3.True : Using the test set is done in order to check the generalization ability of the model. 
                 In case it is used in the cross validation then it would change the models hyperparameters 
                 and lead to a model with poor generalization ability, so it should not be used in cross validation.
        
        4.False : After performing the K-fold cross validation, we choose the single fold which provided the best performance in terms of the smallest error 
                  when testing on the test set and according to this specific fold we determine our hyperparameters.
                  


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**
    No , although adding regulization affects the loss function, where $\lambda$ is hyper parameter 
    chosen after different values, regularization inhances overfitting on our model, but here in the question
    the fried used the test set, which is the wrong set to choose in this case.


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

    1. This approach delivers model that are very much dependant on the data analyzed in the train set,
       thus it may lead to a model which overfits itself to the data that it knows already, and so the generalization
       ability of this model is poor.
       On the other hand K-fold CV validates the model and its performance on new data, that was not used for building the model,
       this provides a model with better generalization ability.
       
    2. The K-fold CV evaluates the model based on the validation set, which normally consists of data that can be found both in the 
       training and the test sets, and so delivers a more general model.
       

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
    As we shrink all values inside W the score differences will become lower, and as we scale up the weights the score differences will all become     higher.
    

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
     according to the most similar image known to it.
     As a result of this we noticed some falsely classified images.

    2. K-nn is a naive algorithm, it classifies the sample according to its K nearest neighbours in its dataset.
       Difference between the two models is that KNN does not use different weights for different neighbours,
       which causes it to be less representative and more naive.


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
    The ideal pattern to see in the residual plot is points on the axis horizontal axis.
    We got a final plot after CV which is much better than the one with the top 5 features, the distances between the 
    samples and the lines are smaller, which means better performance in terms of calssification.
    

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**
    1. We can look at the model as a linear regression with higher dimensions, we can see that the non-linearity in our case is not
       in the model and its parameters, rather in the dataset itself.
        
    2. No, we can't do that, because there are some functions that can't be fit this way, for example:
       $e^{ax}$, $log(bx)$ where a and b are weights.
    
    3. The effect of adding non-linear features to our data is the same as adding more features to our data. 
       It gives our model more dimensions to find the right hyperplane.
       

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**
    1.In the logspace case we get that the effect between jumps of consecutive numbers is negligible, 
      however a change in jumps of 10 times can be seen.
      In other words, this approach focuses on significant changes.
      
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
