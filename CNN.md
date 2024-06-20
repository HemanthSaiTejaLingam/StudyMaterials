<div>
  <h3>Stake Holders</h3>
  <p>If you use CNNs to power a real product in a real-world setting, you are going to interact with several different profiles:</p>
  <ul>
      <li>
          <p>Data Scientist / Machine Learning Engineer: Responsible for developing the ML pipeline and the model, as well as performing all the relevant analytics - for example on data quality and performance measurements.</p>
      </li>
      <li>
          <p>Data Engineers: Responsible for the data ingestion pipelines, the quality of the data that the DS/MLE receive, and for provisioning the right data at inference time.</p>
      </li>
      <li>
          <p>Software Engineers: Responsible for the production environment, both front-end and back-end, as well as for many pieces of the MLOps infrastructure. Involve them from the beginning to make sure that the model you are producing can fit into the production environment.</p>
      </li>
      <li>
          <p>DevOps Engineers: Help in handling the infrastructure, including training servers, various MLOps tools, and other infrastructure needed to train and deploy a model.</p>
      </li>
      <li>
          <p>Product Managers: Define the right problem to solve, exploit the knowledge of the customers, and define quantifiable deliverables and success criteria for the project. The PM also helps in keeping the project on time and on budget.</p>
      </li>
      <li>
          <p>Customers: Consumer of the product; we should always consider the customers' and users' perspectives for every decision that we make.</p>
      </li>
  </ul>
</div>
<div>
  <h2>A brief history of CNNs:</h2>
  <ul>
      <li><a target="_blank" rel="noopener noreferrer" href="https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon">Perceptron (1958)<span>(opens in a new tab)</span></a>: A one-neuron primitive neural network capable of classifying linearly-separable datasets.</li>
      <li><a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Neocognitron#:~:text=The%20neocognitron%20is%20a%20hierarchical,inspiration%20for%20convolutional%20neural%20networks.">Neocognitron (1980)<span>(opens in a new tab)</span></a>: A neural network using two types of mechanisms that are the basis of modern CNNs: convolution and pooling.</li>
      <li><a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Backpropagation#History">Backpropagation (1986)<span>(opens in a new tab)</span></a>: Allows training of neural networks end to end based on data.</li>
      <li><a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Multilayer_perceptron">Multi-Layer Perceptron (1986)<span>(opens in a new tab)</span></a>: The first proper neural network in the modern sense, theoretically capable of modeling any function.</li>
      <li><a target="_blank" rel="noopener noreferrer" href="http://yann.lecun.com/exdb/lenet/">LeNet-5 (1998)<span>(opens in a new tab)</span></a>: The first proper Convolutional Neural Network with practical application, used to model handwritten digits obtaining an accuracy of almost 99%. This seminal work sparked a new renaissance of work on Convolutional Neural Networks for image recognition.</li>
      <li><a target="_blank" rel="noopener noreferrer" href="https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/">ImageNet (2010-2017)<span>(opens in a new tab)</span></a>: A competition with the goal of modeling with the largest possible accuracy a large dataset of more than 1 million natural images classified into 1000 classes.</li>
  </ul>
  <p>The ImageNet competition spawned several innovations, starting with AlexNet, the first CNN to win the contest. AlexNet won in 2012 by a huge margin, when the runners-up were still trying to use classical computer vision methods. After 2012, every team in the competition used Convolutional Neural Networks.</p>
  <p>Since then, the performance on the ImageNet dataset has continued improving at a very fast pace. Most of the architectures from around 2012 - 2020 were based exclusively on Convolutional Neural Networks. After that, a different class of neural network called Transformers started to conquer the top of the rankings. These days the most accurate architectures such as ViT and CoAtNet use a mix of CNN and Transformer elements. Many of these architectures also use additional data.</p>
  <p>CNNs however are still the workhorses for real-world applications of neural networks on images, because they can achieve very good performances while requiring orders of magnitude less data and compute than pure Transformers.</p>
</div>
<div>
    <h2>Definitions</h2>
    <p>Before continuing, let's define some of the terms that we are going to use:</p>
    <ul>
        <li><strong>Vector</strong> - an array of numbers with only one dimension</li>
        <li><strong>Matrix</strong> - an array of numbers with two dimensions</li>
        <li><strong>Array</strong> or tensor - two interchangeable generic terms which can mean arrays with one, two, or n dimensions</li>
    </ul>
    <h2>Computer Interpretation of Images</h2>
    <p>An image is seen by the computer as an array of values (a matrix).</p>
    <p>The images in the MNIST dataset are 28 x 28 and 8-bit grayscale. This means that the computer represents each one of them as a square matrix of 28 x 28 elements, where the value in the element of the matrix represents the light intensity with a range of 0 to 255: 0 means pure black and 255 means pure white.</p>
    <h2>Classification with Neural Networks</h2>
    <p>We already know how to perform classification with neural networks, and in particular with a Multi-Layer Perceptron. This network takes as input a grayscale image (a matrix) and outputs a vector of scores or a vector of probabilities (one for each class). The class corresponding to the maximum of that vector corresponds to the best guess for the label of the image.</p>
    <h2>Flattening</h2>
    <p>Suppose we want to use an MLP to classify our image. The problem is, the network takes a 1d array as input, while we have images that are 28x28 matrices. The obvious solution is to <em>flatten</em> the matrix, i.e., to stack all the rows of the matrix in one long 1D vector, as in the image below.</p>
  <img src="https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/dad16c69-5e2f-4c2b-84ef-00cb48706569">
  <h2>Normalizing Image Inputs</h2>
  <p>Data normalization is an important pre-processing step for neural networks. The activation functions that are normally used in neural networks (sigmoid, ReLU, ...) have the highest rate of change around 0:</p>
  <img src="https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/b8ae2b4e-d7ea-4318-a66d-03c8354a7920">
  <p>This means that their derivative is large for inputs that are not too far from 0. Since we train neural networks with gradient descent, the training can proceed faster if the weights stay in the vicinity of where the activation function changes rapidly, i.e., close to 0.</p>
  <p>The weights of a neural network are typically initialized with a mean of 0, i.e., some weights are negative and some are positive, but they are in general between -1 and +1, close to 0. Remember that these weights are multiplied with the feature values (in this case the pixel values) and then a bias is added on top before the result is fed to the activation function.</p>
  <p>Therefore, if we want the input of the activation function to be somewhere close to 0, we need to start with a number that is close to zero, because two numbers close to zero multiplied together give another number close to zero.</p>
  <p>So we need to take the pixels in the input image, which in the case of a grayscale image have values between 0 and 255, and renormalize them to be close to zero.</p>
  <p>The easiest way is to just divide the value by 255, thereby changing the pixel values to be between 0 and 1.</p>
  <p>In many cases, we go further than that: we compute the mean and the standard deviation of all the pixels in the renormalized dataset, then we subtract the mean and divide by the standard deviation for each image separately. Therefore, our transformed data will contain both negative and positive pixels with mean 0 and standard deviation 1. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values. If you'd like, you can read more here about the <a target="_blank" href="https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html">Normalize transformation in PyTorch</a>.</p>
</div>
<div>
  <h2>Loss Function</h2>
  <p>The <strong>loss function</strong> quantifies how far we are from the ideal state where the network does not make any mistakes and has perfect confidence in its answers.</p>
  <p>Depending on the task and other considerations we might pick different loss functions. For image classification the most typical loss function is the <strong>Categorical Cross-Entropy (CCE) loss</strong>, defined as:</p>
  <div>
    <img src="https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/4aaf74e6-7983-4dc4-83c6-83bdea6587d6">
  </div>
  <p>where:</p>
  <ul>
      <li>The sum is taken over the classes (10 in our case)</li>
      <li>y<sub>i</sub> is the ground truth, i.e., a <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/One-hot#Machine_learning_and_statistics">one-hot encoded vector</a> of length 10</li>
      <li>p<sub>i</sub> is the probability predicted by the network</li>
  </ul>
  <blockquote>
      <p>NOTE: In PyTorch it is customary to have the network output class scores (or <em>logits</em>) instead of probabilities like in other frameworks. Then, the PyTorch loss function itself will first normalize the scores with a Softmax function, and then compute the CCE loss.</p>
  </blockquote>
</div>
<div>
    <h2>Gradient Descent</h2>
    <p>The values of the weights and the biases in the MLP are optimized by using Gradient Descent (or more precisely Stochastic Gradient Descent or SGD for short), or similar algorithms such as <a target="_blank" rel="noopener noreferrer" href="https://optimization.cbe.cornell.edu/index.php?title=Adam">Adam</a>. SGD takes the derivative of the loss with respect to the weights and then updates the values of the weights so as to decrease the loss as quickly as possible.</p>
</div>
