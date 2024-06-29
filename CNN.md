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
<div>
  <h2>Convolution on Color Images</h2>
  <p>The kernel that was a matrix of k x k numbers for grayscale images, becomes now a 3d filter of k x k x n channels:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c7b05268-15b4-462e-8f62-64efb147972c'>
  <h2>Multiple Convolutional Layers</h2>
  <p>In a CNN with more than one layer, the 𝑛_𝑘 filters in the first convolutional layer will operate on the input image with 1 or 3 channels (RGB) and generate 
𝑛_𝑘 output feature maps. So in the case of an RGB image the filters in the first convolutional layer will have a shape of kernel_size x kernel_size x 3. If we have 64 filters we will then have 64 output feature maps. Then, the second convolutional layer will operate on an input with 64 "channels" and therefore use filters that will be kernel_size x kernel_size x 64. Suppose we use 128 filters. Then the output of the second convolutional layer will have a depth of 128, so the filters of the third convolutional layer will be kernel_size x kernel_size x 128, and so on. For this reason, it is common to use the term "channels" also to indicate the feature maps of convolutional layers: a convolutional layer that takes feature maps with a depth of 64 and outputs 128 feature maps is said to have 64 channels as input and 128 as outputs.</p>
  <h2>Number of Parameters in a Convolutional Layer</h2>
  <p>Let's see how we can compute the number of parameters in a convolutional layer,</p>
  <img src="https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c8fcba0f-5ded-497e-a5b1-8c5a6c51eb43">
</div>
<div>
  <h2>Structure of a Typical CNN</h2>
  <p>In a typical CNN there are several convolutional layers intertwined with Max Pooling layers. The convolutional layers have more and more feature maps as you go deeper into the network, but the size of each feature map gets smaller and smaller thanks to the Max Pooling layer.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/2eec5ac6-db21-4bc3-90d2-867f06a9eb45'>
  <p>This kind of structure goes hand in hand with the intuition we have developed in another lesson: as the signal goes deeper into the network, more and more details are dropped, and the content of the image is "abstracted." In other words, while the initial layers focus on the constituents of the objects (edges, textures, and so on), the deeper layers represent and recognize more abstract concepts such as shapes and entire objects.</p>
</div>
<div>
  <h2>Feature Vectors</h2>
  <p>A classical CNN is made of two distinct parts, sometimes called the <strong>backbone</strong> and the <strong>head</strong>, illustrated below.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/2566cda3-6f36-407a-b030-67a9973bfcfe'>
  <p>The <strong>backbone</strong> is made of convolutional and pooling layers, and has the task of extracting information from the image.</p>
  <p>After the backbone there is a flattening layer that takes the output feature maps of the previous convolutional layer and flattens them out in a 1d vector: for each feature map the rows are stacked together in a 1d vector, then all the 1d vectors are stacked together to form a long 1d vector called a <strong>feature vector</strong> or <strong>embedding</strong>. This process is illustrated by the following image:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/3585d9ea-ac5c-4f13-881e-a2ff75483e91'>
  <p>After the flattening operation we have the <strong>head</strong> section. The head is typically just a normal MLP that takes as input the feature vector and has the appropriate output for the task. It can have one or more hidden layers, as well as other types of layers as needed (like DropOut for regularization). In case of a classification task the output dimension is equal to the number of classes, just as in a normal MLP.</p>
</div>
<div>
  <h2>Optimizing the Performance of Our Network</h2>
  <h3>Image Augmentation</h3>
  <p>The basic idea of image augmentation is the following: if you want your network to be insensitive to changes such as rotation, translation, and dilation, you can use the same input image and rotate it, translate it, and scale it and ask the network not to change its prediction!</p>
  <p>In practice, this is achieved by applying random transformations to the input images before they are fed to the network.</p>
  <p>Image augmentation is a very common method to:</p>
  <ol>
    <li>Increase the robustness of the network</li>
    <li>Avoid overfitting</li>
    <li>Introduce rotational, translational and scale invariance as well as insensitiveness to color changes</li>
    <li>Avoid <a href='https://news.mit.edu/2021/shortcut-artificial-intelligence-1102'>shortcut learning</a></li>
  </ol>
  <h3>Batch Normalization</h3>
  <p>The second modern trick that paves the way for enhancing the performance of a network is called <strong>Batch Normalization</strong>, or <strong>BatchNorm</strong>. It does not usually improve the performances per se, but it allows for much easier training and a much smaller dependence on the network initialization, so in practice it makes our experimentation much easier, and allows us to more easily find the optimal solution.</p>
  <h4>How BatchNorm Works</h4>
  <p>Just as we normalize the input image before feeding it to the network, we would like to keep the feature maps normalized, since they are the output of one layer and the input to the next layer. In particular, we want to prevent them to vary wildly during training, because this would require large adjustments of the subsequent layers. Enter BatchNorm. BatchNorm normalizes the activations and keep them much more stable during training, making the training more stable and the convergence faster.</p>
  <p>In order to do this, during training BatchNorm needs the mean and the variance for the activations for each mini-batch. This means that the batch size cannot be too small or the estimates for mean and variance will be inaccurate. During training, the BatchNorm layer also keeps a running average of the mean and the variance, to be used during inference.</p>
  <p>During inference we don't have mini-batches. Therefore, the layer uses the mean and the variance computed during training (the running averages).</p>
  <p>This means that BatchNorm behaves differently during training and during inference. The behavior changes when we set the model to training mode (using <code>model.train()</code>) or to validation mode (<code>model.eval()</code>).</p>
  <h4>Pros and Cons of Batch Normalization</h4>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/9a44ab44-f6d2-43c3-9b27-87847683fa5e'>
  <p>These advantages of using BatchNorm generally outweigh these disadvantages, so BatchNorm is widely used in almost all CNN implementations today.</p>
</div>
<div>
  <h2>Important Terms in Optimizing Performance</h2>
  <p><strong>Parameter</strong></p>
  <ul>
    <li>Internal to the model</li>
    <li>May vary during training</li>
    <li>Examples: Weights and biases of a network</li>
  </ul>
  <p><strong>Hyperparameter</strong></p>
  <ul>
    <li>External to the model</li>
    <li>Fixed during training</li>
    <li>Examples: Learning rate, number of layers, activation layers</li>
  </ul>
  <p><strong>Experiment</strong></p>
  <ul>
    <li>A specific training run with a fixed set of hyperparameters</li>
    <li>Practitioners typically perform many experiments varying the hyperparameters. Each experiment produces one or more metrics that can be used to select the best-performing set of hyperparameters (see the next section).</li>
  </ul>
  <h2>Strategies for Optimizing Hyperparameters</h2>
  <p><strong>Grid search</strong></p>
  <ul>
    <li>Divide the parameter space in a regular grid</li>
    <li>Execute one experiment for each point in the grid</li>
    <li>Simple, but wasteful</li>
  </ul>
  <p><strong>Random Search</strong></p>
  <ul>
    <li>Divide the parameter space in a random grid</li>
    <li>Execute one experiment for each point in the grid</li>
    <li>Much more efficient sampling of the hyperparameter space with respect to grid search</li>
  </ul>
  <p><strong>Bayesian Optimization</strong></p>
  <ul>
    <li>Algorithm for searching the hyperparameter space using a Gaussian Process model</li>
    <li>Efficiently samples the hyperparameter space using minimal experiments</li>
  </ul>
  <h2>Most Important Hyperparameters</h2>
  <p>Optimizing hyperparameters can be confusing at the beginning, so we provide you with some rules of thumb about the actions that typically matter the most. They are described in order of importance below. These are not strict rules, but should help you get started:</p>
  <ol>
    <li>Design parameters: When you are designing an architecture from scratch, the number of hidden layers, as well as the layers parameters (number of filters, width and so on) are going to be important.</li>
    <li>Learning rate: Once the architecture is fixed, this is typically the most important parameter to optimize. </li>
    <li>Batch size: This is typically the most influential hyperparameter after the learning rate. A good starting point, especially if you are using BatchNorm, is to use the maximum batch size that fits in the GPU you are using. Then you vary that value and see if that improves the performances.</li>
    <li>Regularization: Once you optimized the learning rate and batch size, you can focus on the regularization, especially if you are seeing signs of overfitting or underfitting.</li>
    <li>Optimizers: Finally, you can also fiddle with the other parameters of the optimizers. Depending on the optimizers, these vary. Refer to the documentation and the relevant papers linked there to discover what these parameters are.</li>
  </ol>
  <h2>Optimizing Learning Rate</h2>
  <p>The learning rate is one of the most important hyperparameters. However, knowing what the optimal value is, or even what a good range is, can be challenging.</p>
  <p>One useful tool to discover a good starting point for the learning rate is the so-called "learning rate finder." It scans different values of the learning rate, and computes the loss obtained by doing a forward pass of a mini-batch using that learning rate. Then, we can plot the loss vs. the learning rate and obtain a plot similar to this:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/7a34172a-4213-424d-85cf-23f46423be6e'>
  <h3>Learning Rate Schedulers</h3>
  <p>In many cases we want to vary the learning rate as the training progresses. At the beginning of the training we want to make pretty large steps because we are very far from the optimum. However, as we approach the minimum of the loss, we need to make sure we do not jump over the minimum.</p>
  <p>For this reason, it is often a good idea to use a <strong>learning rate scheduler</strong>, i.e., a class that changes the learning rate as the training progresses.</p>
  <p>There are several possible learning rate schedulers. You can find the available ones in the <a href='https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate'>PyTorch learning rate schedulers documentation</a>.</p>
  <p>One of the simplest one is the <code>StepLR</code> scheduler. It reduces the learning rate by a specific factor every <code>n</code> epochs. It can be used as follows:</p>
</div>
<div>
  <h2>Tracking Your Experiments</h2>
  <p>When you are performing hyperparameter optimization and other changes it is very important that you track all of your experiments. This way you will know which hyperparameters have given you which results, and you will be able to repeat those experiments, choose the best one, understand what works and what doesn't, and what you need to explore further. You will also be able to present all your results to other people.</p>
  <p>You can of course use spreadsheets for this, or even pen and paper, but there are definitely much better ways!</p>
  <p>Enter experiment tracking tools. There are many of them out there, and they all work in similar ways. Let's consider <a href='https://www.mlflow.org/docs/latest/tracking.html'>mlflow</a>, which is free and open source.</p>
  <p>Tracking an experiment is easy in mlflow. You first start by creating a run. A run is a unit of execution that will contain your results. Think of it as one row in a hypothetical spreadsheet, where the columns are the things you want to track (accuracy, validation loss, ...). A run can be created like this:</p>
  <pre><code>
with mlflow.start_run():
  ... your code here ...</code>
  </pre>
  <p>Once you have created the run, you can use <code>mlflow.log_param</code> to log a parameter (i.e., one of the hyperparameters for example) and <code>mlflow.log_metric</code> to log a result (for example the final accuracy of your model). For example, let's assume that our only hyperparameters are the learning rate and the batch size. We can track their values as well as the results obtained when using those values like this:</p>
  <pre>
    <code>
import mlflow
with mlflow.start_run():
        ... train and validate ...
    # Track values for hyperparameters    
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    # Track results obtained with those values
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)
    # Track artifacts (i.e. files produced by our experiment)
    # For example, we can save the weights for the epoch with the
    # lowest validation loss
    mlflow.log_artifact("best_valid.pt")</code>
  </pre>
  <p>If we do this for all of our experiments, then <code>mlflow</code> will allow us to easily study the results and understand what works and what doesn't. It provides a UI that looks like this:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/785fbbf7-87b1-4bff-b0ef-7fe83da6fffe'>
  <p>But you can also look at the results in a notebook by doing:
  <pre>
    <code>runs = mlflow.search_runs()</code>
  </pre>
  <p><code>runs</code> is a pandas DataFrame that you can use to look at your results.</p>
  <p>We barely scratched the surface about what a tracking tool like <code>mlflow</code> can do for you. For example, they track the code that runs in your experiment so you can reproduce it even if you changed the code in the meantime. If you are looking to apply what you are learning in this course in a professional environment, have a good look at tracking tools and how they can benefit you.</p>
</p>
</div>
<div>
  <h2>Weight Initialization</h2>
  <h3>What is Weight Initialization?</h3>
  <p>Weight initialization is a procedure that happens only once, before we start training our neural network. Stochastic Gradient Descent and other similar algorithms for minimization are iterative in nature. They start from some values for the parameters that are being optimized and they change those parameters to achieve the minimum in the objective function (the loss).</p>
  <p>These "initial values" for the parameters are set through <strong>weight initialization</strong>.</p>
  <p>Before the introduction of BatchNorm, weight initialization was really key to obtaining robust performances. In this previous era, a good weight initialization strategy could make the difference between an outstanding model and one that could not train at all. These days networks are much more forgiving. However, a good weight initialization can speed up your training and also give you a bit of additional performance.</p>
  <p>In general, weights are initialized with random numbers close but not equal to zero, not too big but not too small either. This makes the gradient of the weights in the initial phases of training neither too big nor too small, which promotes fast training and good performances. Failing to initialize the weights well could result in <a href='https://en.wikipedia.org/wiki/Vanishing_gradient_problem'>vanishing or exploding gradients</a>, and the training would slow down or stop altogether.</p>
  <h3>Weight Initialization in PyTorch</h3>
  <p>By default, PyTorch uses specific weight initialization schemes for each type of layer. This in practice means that you rarely have to think about weight initialization, as the framework does it for you using high-performance default choices.</p>
  <p>If you are curious, you can see how each layer is initialized by looking at the <code>reset_parameters</code> method in the code for that layer. For example, <a target="_blank" href="https://github.com/pytorch/pytorch/blob/f9d07ae6449224bdcb6eb69044a33f0fb5780adf/torch/nn/modules/linear.py#L92">this is the initialization strategy for a Linear layer</a> (i.e., a fully-connected layer), and <a target="_blank" href="https://github.com/pytorch/pytorch/blob/f9d07ae6449224bdcb6eb69044a33f0fb5780adf/torch/nn/modules/conv.py#L140">this is the initialization strategy for a Conv2d layer</a>. In both cases PyTorch uses the so-called <a target="_blank" href="https://paperswithcode.com/method/he-initialization">He initialization</a> (or Kaiming initialization).</p>

</div>
