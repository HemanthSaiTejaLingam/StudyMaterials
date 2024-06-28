<div>
  <p>To create a convolutional layer in PyTorch, you must first import the necessary module:</p>
  <pre>
    <code>from torch import nn</code>
  </pre>
  <p>Then you can define a convolutional layer as:</p>
  <pre>
    <code>conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)</code>
  </pre>
  <p>You must pass the following arguments:</p>
  <ul>
    <li><code>in_channels</code> - The number of input feature maps (also called channels). If this is the first layer, this is equivalent to the number of channels in the input image, i.e., 1 for grayscale images, or 3 for color images (RGB). Otherwise, it is equal to the output channels of the previous convolutional layer.</li>
    <li><code>out_channels</code>  - The number of output feature maps (channels), i.e. the number of filtered "images" that will be produced by the layer. This corresponds to the unique convolutional kernels that will be applied to an input, because each kernel produces one feature map/channel. Determining this number is an important decision to make when designing CNNs, just like deciding on the number of neurons is an important decision for an MLP.</li>
    <li><code>kernel_size</code> - Number specifying both the height and width of the (square) convolutional kernel.</li>
  </ul>
  <p><a target="_blank" href="https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d">official PyTorch documentation</a></p>
  <p>Note in PyTorch that this convolutional layer does NOT include the activation function, which is different than in other deep learning libraries such as Keras. We need to add that as a separate layer.</p>
  <h2>Adding the Activation Function and Dropout</h2>
  <p>As we have seen in the case of MLPs, the activation function must be specified as a separate layer. The same is true for the Dropout layer, which is a very effective regularizer for neural networks. In the case of CNNs, we need to use the 2d version of Dropout, which randomly drops some input channels entirely.</p>
  <p>So, a convolutional block in PyTorch looks something like:</p>
  <pre>
    <code>conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
dropout1 = nn.Dropout2d(p=0.2)
relu1 = nn.ReLU()</code>
  </pre>
  <p>We can now apply it to an input image <code>x</code>:</p>
  <pre>
    <code>result = relu1(dropout1(conv1(x)))</code>
  </pre>
  <p>We can also use <code>nn.Sequential</code>, which stacks together the layers we give as argument so they can be used as if they were one. For example we can build a convolutional block as:</p>
  <pre>
    <code>conv_block = nn.Sequential(
  nn.Conv2d(in_channels, out_channels, kernel_size),
  nn.ReLU(),
  nn.Dropout2d(p=0.2)
)</code>
  </pre>
  <p>and now we can use it simply as:</p>
  <pre>
    <code>result = conv_block(x)</code>
  </pre>
</div>
<div>
  <h2>Padding, Stride, Input and Output Size</h2>
  <p><strong>Padding:</strong> Expanding the size of an image by adding pixels at its border</p>
  <p><strong>Stride:</strong> Amount by which a filter slides over an image.</p>
  <p>In PyTorch, the <code>Conv2d</code> layer allows for an arbitrary amount of padding.</p>
  <p>Let's consider the first convolutional layer in a CNN, and let's assume that the input images are 5x5x1 (height of 5, width of 5, and grayscale) to make our math easier.</p>
  <p>We can, for example, define a first convolutional layer with 1 input channel (corresponding to the intensity of the image), 16 feature maps (also called output channels), a 3x3 kernel, and stride 1. This is defined in PyTorch as <code>nn.Conv2d(1, 16, kernel_size=3, padding=0)</code>code>. We can visualize how one filter operates with the following animation:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/baa47c9f-8e09-42de-9b4e-ef4c6d52ceb0'>
  <p>As we can clearly see, the kernel fits in the image only 3 times per row, thus the output feature map is 3x3.</p>
  <p>In many cases it is a good idea to keep the input size and the output size the same. Sometimes it is even necessary, as we will see in a different lesson when we talk about skip connections. In this particular case, we just need to add a padding of 1: <code>nn.Conv2d(3, 16, kernel_size=3, padding=1)</code> (we will see later the different strategies to fill those values).</p>
  <p>This is the result:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/f65dd8dd-3529-43e8-a7ce-221e30466ca4'>
  <p>What happens if instead of a 3x3 kernel we consider a 5x5 kernel? Obviously, without padding, the kernel would only fit once in the image. If we add a padding of 1, the kernel will fit 9 times in total and produce an image that is 3x3.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/820ccb36-5c39-4da2-a0b6-93b5e09fd1ad'>
  <p>If we want our 5x5 convolution to produce the same size as the input, we have to use a padding of 2 this time:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/6b08e531-2438-43b1-bc23-cf556d12a262'>
  <h2>Formula for Convolutional Layers</h2>
  <p>n general we can link together the output size o, the size of the input image i, the size of the kernel k, the stride s , and the padding p with this simple formula, which is very useful when you are creating your own architectures:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/eb728517-2094-4874-b411-d42c7ff10d30'>
  <h4>PyTorch Shortcuts</h4>
  <p>In PyTorch you can also use <code>padding="same"</code> or <code>padding="valid"</code> instead of providing explicit numbers for the padding. The <code>same</code> padding instructs PyTorch to automatically compute the amount of padding necessary to give you an output feature map with the same shape as the input. Note this option only works if you are using a stride of 1. The <code>valid</code> padding is equivalent to using a padding of 0.</p>
  <h4>Types of Padding</h4>
  <p>There are different strategies to fill the padding pixels. The most obvious one is to fill them with zero (the default in PyTorch). However, the <code>nn.Conv2d</code> layer in Pytorch also supports other options, illustrated by the following images:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/94267f0e-6fd7-4072-a2ca-5ec37a825de9'>
    <p><code>padding_mode="reflect"</code>: the padding pixels are filled with a copy of the values in the input image taken in opposite order,<br> in a mirroring fashion.</p>
  </pre>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/6d331512-dcdc-4449-8a39-993a59fbdc1e'>
    <p><code>padding_mode="replicate"</code>: the padding pixels are filled with value of closest pixel in input image.</p>
  </pre>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/7d9bbe3c-e508-4b65-a5d9-226a0a9e3856'>
    <p><code>padding_mode="circular"</code>: it is like the reflection mode, but the image is first flipped horizontally and vertically.</p>
  </pre>
  <p>The zero-padding strategy is by far the most common, but these other strategies are adopted in some architectures and you can definitely experiment with them to see if they make any difference with respect to the zero-padding.</p>
  <p><strong>Average Pooling</strong> is not typically used for image classification problems because Max Pooling is better at noticing the most important details about edges and other features in an image, but you may see average pooling used in applications for which smoothing an image is preferable.</p>
  <p>Sometimes, Average Pooling and Max Pooling are used together to extract both the maximum activation and the average activation.</p>
  <h2>Pooling Layers in PyTorch</h2>
  <p>To create a pooling layer in PyTorch, you must first import the necessary module:</p>
  <pre><code>from torch import nn</code></pre>
  <p>Then you can define the layer as:</p>
  <pre><code>nn.MaxPool2d(kernel_size, stride)</code></pre>
  <ul>
    <li><code>kernel_size</code> - The size of the max pooling window. The layer will roll a window of this size over the input feature map and select the maximum value for each window.</li>
    <li><code>stride</code> - The stride for the operation. By default the stride is of the same size as the kernel (i.e., kernel_size).</li>
  </ul>
  <p>There are some additional optional arguments that are rarely used, but could be helpful under certain circumstances. Please refer to the <a target="_blank" href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html">PyTorch documentation on MaxPool2D</a>.</p>
  <p>Similarly, an Average Pooling Layer can be created in this way:</p>
  <pre><code>pooling = nn.AvgPool2d(window_size, stride)</code></pre>
</div>
