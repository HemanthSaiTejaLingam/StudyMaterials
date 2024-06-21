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
<div>
  <h3>Classifier Performance (MLP VS CNN)</h3>
  <p>The MNIST dataset is very clean and is one of the few datasets where MLPs and Convolutional Neural Networks perform at a similar level of accuracy. However, all of the <a target="_blank" rel="noopener noreferrer" href="https://paperswithcode.com/sota/image-classification-on-mnist">top-scoring architectures for MNIST<span>(opens in a new tab)</span></a> are CNNs (although their performance difference compared to MLPs is small).</p>
  <p>In most cases, CNNs are vastly superior to MLPs, both in terms of accuracy and in terms of network size when dealing with images.</p>
  <p>As we will see, the main reason for the superiority of CNNs is that MLPs have to flatten the input image, and therefore initially ignore most of the spatial information, which is very important in an image. Also, among other things, they are not invariant for translation. This means that they need to learn to recognize the same image all over again if we translate even slightly the objects in it.</p>
  <p>CNNs instead don't need to flatten the image and can therefore immediately exploit the spatial structure. As we will see, through the use of convolution and pooling they also have approximate translation invariance, making them much better choices for image tasks.</p>
</div>
<div>
  <h2>Locally-Connected Layers</h2>
  <p>Convolutional Neural Networks are characterized by locally-connected layers, i.e., layers where neurons are connected to only a limited numbers of input pixels (instead of all the pixels like in fully-connected layers). Moreover, these neurons share their weights, which drastically reduces the number of parameters in the network with respect to MLPs. The idea behind this weight-sharing is that the network should be able to recognize the same pattern anywhere in the image</p>
</div>
<div>
  <h2>The Convolution Operation</h2>
  <p>CNNs can preserve spatial information, and the key to this capability is called the Convolution operation: it makes the network capable of extracting spatial and color patterns that characterize different objects.</p>
  <p>CNNs use <strong>filters</strong> (also known as <strong>"kernels"</strong>) to "extract" the features of an object (for example, edges). By using multiple different filters the network can learn to recognize complex shapes and objects.</p>
</div>
<div>
  <h2>Image Filters</h2>
  <p><strong>Image filters</strong> are a traditional concept in computer vision. They are small matrices that can be used to transform the input image in specific ways, for example, highlighting edges of objects in the image.</p>
  <p>An edge of an object is a place in an image where the intensity changes significantly.</p>
  <p>To detect these changes in intensity within an image, you can create specific image filters that look at groups of pixels and react to alternating patterns of dark/light pixels. These filters produce an output that shows edges of objects and differing textures</p>
</div>
<div>
  <h2>Frequency in Images</h2>
  <p>We have an intuition of what frequency means when it comes to sound. A high frequency of vibration produces a high-pitched noise, like a bird chirp or violin. And low-frequency vibrations produce low pitches, like a deep voice or a bass drum. For sound, frequency actually refers to how fast a sound wave is oscillating; oscillations are usually measured in cycles/s (Hertz or Hz(opens in a new tab)), and high pitches are created by high-frequency waves. Examples of low- and high-frequency sound waves are pictured below. On the y-axis is amplitude, which is a measure of sound pressure that corresponds to the perceived loudness of a sound, and on the x-axis is time.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/0099ce04-2e74-4795-82df-19aae0f8df80'>
  <h6>High and Low Frequency</h6>
  <p>Similarly, frequency in images is a <strong>rate of change</strong>. But, what does it means for an image to change? Well, images change in space, and a high-frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low-frequency image may be one that is relatively uniform in brightness or changes very slowly. This is easiest to see in an example.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/83d96a4f-d874-445e-bcda-7c18dfae4960'>
  <p>Most images have both high-frequency and low-frequency components. In the image above, on the scarf and striped shirt, we have a high-frequency image pattern; this part changes very rapidly from one brightness to another. Higher up in this same image, we see parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern.</p>
  <p><strong>High-frequency components also correspond to the edges of objects in images</strong>, which can help us classify those objects.</p>
</div>
<div>
  <h2>Edge Handling</h2>
  <p>Kernel convolution relies on centering a pixel and looking at its surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.</p>
  <p>strong>Padding</strong> - The image is padded with a border of 0's, black pixels.</p>
  <p>strong>Cropping</strong> - Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being smaller then the input image, with the edges having been cropped.</p>
  <p>strong>Extension</strong> - The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.</p>
</div>
<div>
  <h2>Pooling</h2>
  <p><strong>Pooling</strong> is a mechanism often used in CNNs (and in neural networks in general). Pooling compresses information from a layer by summarizing areas of the feature maps produced in that layer. It works by sliding a window over each feature map, just like convolution, but instead of applying a kernel we compute a summary statistic (for example the maximum or the mean). If we take the maximum within each window, then we call this <strong>Max Pooling</strong>.</p>
  <h3>Concept Abstraction and Translation Variance</h3>
  <p>A block consisting of a convolutional layer followed by a max pooling layer (and an activation function) is the typical building block of a CNN.</p>
  <p>By combining multiple such blocks, the network learns to extract more and more complex information from the image.</p>
  <p>Moreover, combining multiple blocks allows the network to achieve translation invariance, meaning it will be able to recognize the presence of an object wherever that object is translated within the image.</p>
</div>
<div>
  <h2>Effective Receptive Field</h2>
  <p>The concept of <strong>receptive field</strong> is that a pixel in the feature map of a deep layer is computed using information that originates from a large area of the input image, although it is mediated by other layers:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/1a701a0f-f877-4ec9-a981-a5d91cd970d5'>
<h2>Going Deeper</h2>
  <p>In practice things are a bit more complicated. When we compute the effective receptive field, instead of considering just whether the information contained in a given pixel is used or not by a pixel in a deeper layer, we can consider how many times that pixel is used. In other words, how many times that pixel was part of a convolution that ended up in a result used by the pixel in the deeper layer. Of course, pixels on the border of the input image are used during fewer convolutions than pixels in the center of the image. We can take this even further and ask how much a given pixel in the input image influences the pixel in a feature map deeper in the network. This means, if we change the value of the input pixel slightly, how much does the pixel in the deep layer change. If we take this into account, we end up with receptive fields that are more Gaussian-like, instead of flat, and they also evolve as we train the network.</p>
  <p>To explore this topic further, look at <a target="_blank" href="https://arxiv.org/abs/1701.04128">a paper: Understanding the Effective Receptive Field in Deep Convolutional Neural Networks by Wenjie Luo et al.</a> This is an example taken from it:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/d7909575-eeab-4f6a-92b9-184f000490ab'>
</div>
<div>
  <h2>CNN structure</h2>
  <p>We now have all the elements to see at a high level the structure of a typical CNN.</p>
  <h2>Convolution and Pooling Layers</h2>
  <p>We have building blocks made by convolutional layers followed by pooling layers. We stack these blocks one after the other to build more and more abstraction in order to recognize the object of interest.</p>
  <h2>Flattening</h2>
  <p>Then, we end up with a large number of feature maps that are numerous but smaller than the input image, because of all the pooling operations. We flatten them out in a long array. All the activated pixels in the last feature maps before the flattening become large values in this flattened array.</p>
  <h2>Multi-Layer Perceptron</h2>
  <p>We then have a standard MLP that takes as input the flattened array, and returns as output the class scores.</p>
  <p>We can see here that the convolution part is used to extract features in the form of activated pixels in the flattened array, and then the MLP decides the class based on those features.</p>
  <p>For example, if the convolutional part of the network detects a large square, a small square and a large triangle, then the MLP will use this information to recognize that the image contains a house:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/46b00673-d15e-4985-afad-c70d9593a78b'>
  <p><strong>Sobel filters</strong>: Specific types of filters that can isolate vertical and horizontal edges.</p>
  <p>Here is the website shown in the video, where you can go and explore yourself:</p>
  <p><a target="_blank" href="https://www.cs.cmu.edu/~aharley/vis/conv/flat.html">2D convolutional network visualization</a></p>
  <p>There is also a <a target="_blank" href="https://www.cs.cmu.edu/~aharley/vis/conv/">3D version</a> as well as versions for MLPs (<a target="_blank" href="https://www.cs.cmu.edu/~aharley/vis/fc/flat.html">2D</a> and <a target="_blank" rel="noopener noreferrer" href="https://www.cs.cmu.edu/~aharley/vis/fc/">3D<span>(opens in a new tab)</span></a>).</p>
  <p>These visualizations are described in the paper: <a target="_blank" href="https://www.cs.cmu.edu/~aharley/vis/">A. W. Harley, "An Interactive Node-Link Visualization of Convolutional Neural Networks," in ISVC, pages 867-877, 2015</a></p>
  <h2>External Resource</h2>
  <p><a target="_blank" href="https://www.deeplearningbook.org/">Deep Learning eBook</a>(2016) authored by Ian Goodfellow, Yoshua Bengio, and Aaron Courville; published by Cambridge: MIT Press. This is a terrific free resource!</p>
</div>
