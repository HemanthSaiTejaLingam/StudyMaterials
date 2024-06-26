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
