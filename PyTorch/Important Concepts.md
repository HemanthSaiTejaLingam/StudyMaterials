<div>
  <h2>Dividing Data</h2>
  <p>When presented with a dataset, if we use the whole thing to train our model, then we do not know how it performs on unseen data. We typically divide our data into three sets whose size can vary, but a good rule of thumb is the 80/10/10 rule:</p>
  <ul>
    <li>Train (80%)</li>
    <li>Validation (10%)</li>
    <li>Test (10%)</li>
  </ul>
  <p>Another powerful approach is k-fold cross-validation, where the data is split up into some number, which we call k, equal parts. One is used as the validation set, one is used as the test set, and the remaining parts are used as the training set. We then cycle through all combinations of the data until all parts have had a chance to be the test set.</p>
</div>
<div>
  <p>When we train our models, it is entirely possible to get them to a point where they perform very well on our training data—but then perform very poorly on our testing data. Two common reasons for this are underfitting and overfitting</p>

  <h2>Underfitting</h2>
  <ul>
    <li>Underfitting means that our model is too simplistic. There is a poor fit between our model and the data because we have <b>oversimplified</b> the problem.</li>
    <li>Underfitting is sometimes referred to as <b>error due to bias</b>. Our training data may be biased and this bias may be incorporated into the model in a way that oversimplifies it.</li>
  </ul>
  <p>For example, suppose we train an image classifier to recognize dogs. And suppose that the only type of animal in the training set is a dog. Perhaps the model learns a biased and overly simple rule like, "if it has four legs it is a dog". When we then test our model on some data that has other animals, it may misclassify a cat as a dog—in other words, it will underfit the data because it has error due to bias.</p>
  <h2>Overfitting</h2>
  <ul>
    <li>Overfitting means that our model is too complicated. The fit between our model and the training data is <b>too specific</b>—the model will perform very well on the training data but will <b>fail to generalize</b> to new data.</li>
    <li>Overfitting is sometimes referred to as <b>error due to variance</b>. This means that there are random or irrelevant differences among the data points in our training data and we have fit the model so closely to these irrelevant differences that it performs poorly when we try to use it with our testing data.</li>
  </ul>
  <p>For example, suppose we want our image classifier to recognize dogs, but instead we train it to recognize "dogs that are yellow, orange, or grey." If our testing set includes a dog that is brown, for example, our model will put it in a separate class, which was not what we wanted. Our model is too specific—we have fit the data to some unimportant differences in the training data and now it will fail to generalize.</p>
  <h2>Applying This to Neural Networks</h2>
  <p>Generally speaking, underfitting tends to happen with neural networks that have overly simple architecture, while overfitting tends to happen with models that are highly complex.</p>
  <p>The bad news is, it's really hard to find the right architecture for a neural network. There is a tendency to create a network that either has overly simplistic architecture or overly complicated architecture. In general terms, the approach we will take is to err on the side of an overly complicated model, and then we'll apply certain techniques to reduce the risk of overfitting.</p>
</div>
<div>
  <h2>Early Stopping</h2>
  <p>When training our neural network, we start with random weights in the first epoch and then change these weights as we go through additional epochs. Initially, we expect these changes to improve our model as the neural network fits the training data more closely. But after some time, further changes will start to result in overfitting.</p>
  <p>We can monitor this by measuring both the <b>training error</b> and the <b>testing error</b>. As we train the network, the training error will go down—but at some point, the testing error will start to increase. This indicates overfitting and is a signal that we should stop training the network prior to that point. We can see this relationship in a <b>model complexity graph</b> like this one:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/2e510be0-7940-4df7-9974-5427b2374508'>
  <p>Have a look at the graph and make sure you can recognize the following:</p>
  <ul>
    <li>On the Y-axis, we have a measure of the error and on the X-axis we have a measure of the complexity of the model (in this case, it's the number of epochs).</li>
    <li>On the left we have high testing and training error, so we're underfitting.</li>
    <li>On the right, we have high testing error and low training error, so we're overfitting.</li>
    <li>Somewhere in the middle, we have our happy Goldilocks point (the point that is "just right").</li>
  </ul>
  <p>In summary, we do gradient descent until the testing error stops decreasing and starts to increase. At that moment, we stop. This algorithm is called <b>early stopping</b> and is widely used to train neural networks.</p>
</div>
<div>
  <h2>How to Use Tensorboard</h2>
  <p>After launching Tensorboard from the command line and choosing a directory for the logs, we use the <b>SummaryWriter</b> class from <b>torch.utils.tensorboard</b>. Using the <b>add_scalar</b> method, we can write things like loss and accuracy. We can also put images and figures into Tensorboard using <b>add_image</b> and <b>add_figure</b> respectively.</p>
  <p>For further information, check the<a href='https://pytorch.org/docs/stable/tensorboard.html'>PyTorch Tensorboard documentation</a></p>
</div>
<div>
  <h2>Regularization</h2>
  <h5>Considering the Activation Functions</h5>
  <p>A key point here is to consider the activation functions of these two equations:</p>
  <img src=''>![image](https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/96a1b0c3-72c6-4503-b64d-3f639f26ae20)
  <ul>
    <li>When we apply sigmoid to small values such as 𝑥1+𝑥2, we get the function on the left, which has a nice slope for gradient descent.</li>
    <li>When we multiply the linear function by 10 and take sigmoid of 10𝑥1+10𝑥2, our predictions are much better since they're closer to zero and one. But the function becomes much steeper and it's much harder to do graident descent.</li>
  </ul>
  <p>Conceptually, the model on the right is too certain and it gives little room for applying gradient descent. Also, the points that are classified incorrectly in the model on the right will generate large errors and it will be hard to tune the model to correct them.</p>
  <p>Now the question is, how do we prevent this type of overfitting from happening? The trouble is that large coefficients are leading to overfitting, so what we need to do is adjust our error function by, essentially, penalizing large weights.
<br>
If you recall, our original error function looks like this:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/6913925f-d5dd-4c00-a132-f7aefb0be843'>
<p>We want to take this and add a term that is big when the weights are big. There are two ways to do this. One way is to add the sums of absolute values of the weights times a constant lambda:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/51b3b215-92c2-47df-97f9-d2067fb082cc'>
  <p>The other one is to add the sum of the squares of the weights times that same constant:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/99a84cbc-ae49-4b1a-9479-f8b75a5d1965'>
</div>
<div>
  <h2>L1 vs L2 Regularization</h2>
  <p>The first approach (using absolute values) is called <b>L1 regularization</b>, while the second (using squares) is called <b>L2 regularization</b>. Here are some general guidelines for deciding between the two:</p>
  <h3>L1 Regularization</h3>
  <ul>
    <li>L1 tends to result in sparse vectors. That means small weights will tend to go to zero.</li>
    <li>If we want to reduce the number of weights and end up with a small set, we can use L1.</li>
    <li>L1 is also good for feature selection. Sometimes we have a problem with hundreds of features, and L1 regularization will help us select which ones are important, turning the rest into zeroes.</li>
  </ul>
  <h3>L2 Regularization</h3>
  <ul>
    <li>L2 tends not to favor sparse vectors since it tries to maintain all the weights homogeneously small.</li>
    <li>L2 gives better results for training models so it's the one we'll use the most.</li>
  </ul>
</div>
<div>
  <h2>Dropout</h2>
  <h3>Turning off Weights to Balance Training</h3>
  <p>When training a neural network, sometimes one part of the network has very large weights and it ends up dominating the training, while another part of the network doesn't really play much of a role (so it doesn't get trained).</p>
  <p>To solve this, we can use a method called <b>dropout</b> in which we turn part of the network off and let the rest of the network train:</p>
  <ul>
    <li>We go through the epochs and randomly turn off some of the nodes. This forces the other nodes to pick up the slack and take a larger part in the training.</li>
    <li>To drop nodes, we give the algorithm a parameter that indicates the probability that each node will get dropped during each epoch. For example, if we set this parameter to 0.2, this means that during each epoch, each node has a 20% probability of being turned off.</li>
    <li>Note that some nodes may get turned off more than others and some may never get turned off. This is OK since we're doing it over and over; on average, each node will get approximately the same treatment.</li>
  </ul>
</div>
