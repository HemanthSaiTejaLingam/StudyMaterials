<h2>Origins of the Term Neural Network</h2>
<div>
  <p>Neural Networks get their name from the fact that they are—loosely—modeled after biological neurons. Perceptrons take inputs, perform calculations on the inputs, and decide whether to return one result or another (e.g., a one or a zero).</p>
  <p>In a similar way, neurons in the brain receive inputs (such as signals from other neurons) through their branching dendrites, and then decide whether to, in turn, send out a signal of their own.</p>
  <p>Similar to how real neurons can be connected one to another to form layers, we will be concatenating our perceptrons—layering multiple perceptrons such that we can take the output from one and use it as the input for another.</p>
</div>
<h2>Multilayer Perceptrons are Neural Networks</h2>
<div>
  <p>The perceptron and neural networks are inspired by biological neurons. Though modern "perceptrons" use the Logistic Sigmoid Function or other activation functions, classical perceptrons use a step function.</p>
  <p>Neural Networks are a more general class of models that encapsulates multi-layer perceptrons. Neural Networks are defined by having one or more hidden layers and an output layer that emits a decision -- either a predicted value, a probability, or a vector of probabilities, depending on the task.</p>
  <p><img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c5ff831a-5690-4ca0-bc4e-5294c7b9d84a'></p>
</div>
<div>
  <h2>Combining Models</h2>
<p>We will combine two linear models to get our non-linear model. Essentially the steps to do this are:</p>
<ul>
  <li>Calculate the probability for each model</li>
  <li>Apply weights to the probabilities</li>
  <li>Add the weighted probabilities</li>
  <li>Apply the sigmoid function to the result</li>
</ul>
</div>
<div>
  <p>Neural networks have a certain special architecture with layers:</p>
  <ul>
    <li>The first layer is called the <b>input layer</b>, which contains the inputs.</li>
    <li>The next layer is called the <b>hidden layer</b>, which is the set of linear models created with the input layer.</li>
    <li>The final layer is called the <b>output layer</b>, which is where the linear models get combined to obtain a nonlinear model.</li>
  </ul>
  <p>Neural networks can have different architectures, with varying numbers of nodes and layers:</p>
  <ul>
    <li><b>Input nodes.</b> In general, if we have 𝑛 nodes in the input layer, then we are modeling data in n-dimensional space (e.g., 3 nodes in the input layer means we are modeling data in 3-dimensional space).</li>
    <li><b>Output nodes.</b> If there are more nodes in the output layer, this simply means we have more outputs—for example, we may have a multiclass classification model.</li>
    <li><b>Layers.</b> If there are more layers then we have a deep neural network. Our linear models combine to create nonlinear models, which then combine to create even more nonlinear models!</li>
  </ul>
  <p>When we have three or more classes, we could construct three separate neural networks—one for predicting each class. However, this is not necessary. Instrad, we can add more nodes in the output layer. Each of these nodes will give us the probability that the item belongs to the given class.</p>
</div>
<div>
  <p><b>Feedforward</b> is the process neural networks use to turn the input into an output. In general terms, the process looks like this:
  <ul>
    <li>Take the input vector</li>
    <li>Apply a sequence of linear models and sigmoid functions</li>
    <li>Combine maps to create a highly non-linear map</li>
  </ul>
The feedforward formula is:<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/7343031e-bf68-4682-a10a-46c494a7eeb0'>

</p>
</div>
<div>
  <h2>Activation Function Properties</h2>
  <p>There are a wide variety of activation functions that we can use. Activation functions should be:</p>
  <ul>
    <li>Nonlinear</li>
    <li>Differentiable -- preferably everywhere</li>
    <li>Monotonic</li>
    <li>Close to the identity function at the origin</li>
  </ul>
  <p>We can loosen these restrictions slightly. For example, ReLU is not differentiable at the origin. Others, like monotonicity, are very important and cannot be reasonably relaxed.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/55415a9e-9c43-4a72-a204-a787041bf930'>
</div>
<div>
  <h2>How to Choose an Output Function</h2>
  <p>Your choice of output function depends on two primary factors about what you are trying to predict: its shape and its range.</p>
  <p>This means that our output function is determined by what we're trying to do: classification or regression.</p>
  <p>Common output functions include:</p>
  <ul>
    <li>Sigmoid for binary classification</li>
    <li>Softmax for multi-class classification</li>
    <li>Identity or ReLU for regression</li>
  </ul>
</div>
