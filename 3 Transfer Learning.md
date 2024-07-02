<div>
  <h3>Overview</h3>
  <p>Modern deep neural networks are data-hungry. They require very large datasets, with millions of items, to reach their peak performance.</p>
  <p>Unfortunately, developing such large datasets from scratch for every use case of deep learning is very expensive and often not feasible.</p>
  <p>Transfer learning is a technique that allows you to take a neural network that has already been trained on one of these very large datasets, and tweak it slightly to adapt it to a new dataset.</p>
  <p>This requires far less data than training from scratch. This is the reason why transfer learning is a much more common technique for real-life applications than training from scratch.</p>
  <p>Using a pre-designed and pre-trained architecture instead of designing your own gives you most of the time the best results, and also saves a lot of time and experimentation.</p>
  <p>For all of these reasons, transfer learning is a technique of paramount importance for real-life uses of AI.</p>
</div>
<div>
  <h4>AlexNet</h4>
  <p>The first CNN architecture to use the ReLU activation function. AlexNet also used DropOut to prevent overfitting. It has the structure of a classical CNN, with a backbone made of convolution and Max Pooling followed by a flattening and a Multi-Layer Perceptron.</p>
  <p>You can see the PyTorch code in the <a target="_blank" href="https://github.com/pytorch/vision/blob/59c4de9123eb1d39bb700f7ae7780fb9c7217910/torchvision/models/alexnet.py#L17">AlexNet model implementation on GitHub</a>.</p>
  <h4>VGG</h4>
  <p>This architecture was designed by the Visual Geometry Group at Oxford. There are two versions, VGG16 and VGG19, with 16 and 19 layers respectively. The designers pioneered the use of many 3 by 3 convolutions instead of fewer larger kernels (for example, the first layer of AlexNet uses a 11 by 11 convolution). Most CNNs up to today still use the same strategy. Apart from that, VGG has an elegant and regular architecture made of convolutional layers followed by Max Pooling layers. The height and width of the feature maps decreases as we go deeper into the network, thanks to the Max Pooling layers, but the number of feature maps increases. The backbone is then followed by a flattening operation and a regular head made of a Multi-Layer Perceptron.</p>
  <h4>ResNet in Depth</h4>
  <p>ResNet is a very important architecture that introduced a fundamental innovation: the <strong>skip connection</strong>.</p>
  <p>Before ResNet, deep learning models could not go very deep in terms of number of layers. Indeed, after a certain point, going deeper was hurting performances instead of helping them.</p>
  <p>This pointed to problems in the optimization algorithm, because a deeper network should have at worst an identical performance to a shallower network. Indeed, the optimizer could transform the additional layers into the identity function and recover the shallower network exactly.</p>
  <p>The fact that this does not happen means that the optimizer has a hard time transforming the last layers in the identity function, and so it converges to a suboptimal solution that makes the second network WORSE than the first. This is largely due to the so-called <a target="_blank" href="http://neuralnetworksanddeeplearning.com/chap5.html">vanishing gradient</a> problem.</p>
  <p>How does ResNet solve this problem? By starting very close to the identity function, using the skip connection:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/037761d6-9deb-46be-be08-dcf553739448'>
    <p>In the ResNet block we have two convolutional layers with a ReLU in the middle. The output of these two layers is summed to the input tensor x and then another ReLU is applied on the result.</p>
  <p>This means that the central part comprising the two layers with the ReLU in the middle is learning the residual, from which comes the name Residual Network, or ResNet for short.</p>
  <p>It is easy to see how this block can become the identity function: it is sufficient to put the weights of the kernel of the first or the second convolutional layer to zero (or very close to zero). This will produce a feature map after the two convolutional layers where each pixel is zero. This is then summed to x, which means our block behaves as the identity function because H(x) = x.</p>
  <p>With this simple trick we can now go very deep (up to hundreds of layers) and increase significantly the performance of the network.</p>
  <p>We can implement the ResNet block in PyTorch as follows:</p>
  <pre>
    <code>
class ResidualBlock(nn.Module):
    def __init__(self, inp, out1, out2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(inp, out1, 3),
            nn.ReLU(),
            nn.Conv2d(out1, out2, 3)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        # F(x)
        F = self.conv_block(x)
        # IMPORTANT BIT: we sum the result of the
        # convolutions to the input image
        H = F + x
        # Now we apply ReLU and return
        return self.relu(H)</code>
  </pre>
  <h4>Optional Resources to Explore Further</h4>
    <ul>
      <li>Check out the <a target="_blank" href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">AlexNet</a> paper!</li>
      <li>Read more about <a target="_blank" href="https://arxiv.org/pdf/1409.1556.pdf">VGGNet</a> here.</li>
      <li>The <a target="_blank" href="https://arxiv.org/pdf/1512.03385v1.pdf">ResNet<span>(opens in a new tab)</span></a> paper can be found here.</li>
      <li>Read this <a target="_blank" href="http://neuralnetworksanddeeplearning.com/chap5.html">detailed treatment</a> of the vanishing gradients problem.</li>
      <li>Visit the <a target="_blank" href="http://www.image-net.org/challenges/LSVRC/">ImageNet Large Scale Visual Recognition Competition (ILSVRC)</a> website.</li>
    </ul>
</div>
<div>
  <h2>Fixed Input Size and Global Average Pooling (GAP)</h2>
  <h3>Size of the Input Image for CNNs</h3>
  <p>A classic CNN has a first section comprised of several layers of convolutions and pooling, followed by a flattening and then one or more fully-connected layers.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/de4b2dee-1cd1-472e-a3e7-85501337908f'>
  <p>Convolutional and pooling layers can handle any input size (they will just produce outputs of different size depending on the input size). However, fully-connected layers can only work with an input array of a specific size. Therefore, the vector produced by the flattening operation must have a specific number of elements, because it feeds into the fully-connected layers.</p>
  <p>Let's call this number of elements H. This means that the feature maps that are being flattened must have a specific size, so that n_channels x height x width = H. Since the height and width of the last feature maps are determined by the size of the input image, as it flows through the convolutional and the pooling layers, this constraint on the vector produced by the flattening operation translates to a constraint on the size of the input image. Therefore, for CNNs using flattening layers, the input size must be decided a priori when designing the architecture.</p>
  <h3>Global Average Pooling (GAP) Layer</h3>
  <p>We can now introduce a new pooling layer that is widely used in modern CNNs. This type of pooling is equivalent to average pooling, but the average is taken over the entire feature map. It is equivalent to an Average Pooling Layer with the window size equal to the input size.</p>
  <p>This layer becomes very interesting because it can be used in place of the flattening operation at the end of the convolutional part of a CNN. Instead of taking the last feature maps and flattening them into a long vector, we take the average of each feature map and place them in a much shorter vector:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/5170cd89-6460-489c-8113-1612f98b8f13'>
  <p>This drastically reduces the dimensionality of the resulting vector, from n_channels x height x width to just n_channels. But also, more importantly, it makes the network adaptable to any input size! Let's see how.</p>
  <p>If we use the GAP layer instead of flattening, we are going to obtain a vector of constant size independent of the size of the input image, because the size of the vector after the GAP layer is given by the number of feature maps in the last convolutional layer, and it is not influenced by their height and width. Therefore, the input image can have any size because this will not influence the number of feature maps, but only their height and width.</p>
  <p>Note however that a network with GAP trained on a certain image size will not respond well to drastically different image sizes, even though it will output a result. So effectively the input size became a tunable parameter that can be changed without affecting the architecture of the CNN.</p>
  <p>Many modern architectures adopt the GAP layer.</p>
</div>
<div>
  <h2>Attention Layers</h2>
  <h3>Attention</h3>
  <p>The concept of attention is a very important concept in modern neural networks. It is a simple idea: the network should learn to boost some information that is helpful for a given example, and decrease the importance of information that is not useful for that example.</p>
  <p>There are several forms of attention. Let's look at two important ones.</p>
  <h3>Channel Attention: Squeeze and Excitation</h3>
  <p>The term "channel" can refer to the channels in the input image (3 channels if RGB) but also to the number of feature maps are output from a layer.</p>
  <p>Channel attention is a mechanism that a network can use to learn to pay more attention (i.e., to boost) feature maps that are useful for a specific example, and pay less attention to the others.</p>
  <p>This is accomplished by adding a sub-network that given the feature maps/channels assigns a scale to each input feature map. The feature maps with the largest scale are boosted:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/3d39c741-2ff3-4b6f-9957-2a350778c478'>
  <h2>Brief Introduction to Transformers in Computer Vision</h2>
  <p>Vision Transformers have been recently introduced, and are becoming more and more important for computer vision. They contain another form of attention, called <strong>self attention</strong>.</p>
  <p>Transformers are a family of neural networks originally developed for Natural Language Processing (NLP) applications. They are very good at modeling sequences, such as words in a sentence. They have been extended to deal with images by transforming images to sequences. In short, the image is divided in patches, the patches are transformed into embedded representations, and these representations are fed to a Transformer that treats them as a sequence.</p>
  <p>Transformers are characterized by the self-attention mechanism. Just like channel attention allows the network to learn to focus more on some channels, self attention allows the network to learn how to pay attention to the relationship between different words in a sentence or parts of an image.</p>
  <p>While CNNs build large Effective Receptive Fields by using many layers, vision Transformers show large receptive fields earlier and more consistently throughout the network.</p>
  <h3>Additional Resources</h3>
  <ul>
    <li>Introduction to transformers:
      <ul>
        <li><a target="_blank" href="https://www.youtube.com/watch?v=dichIcUZfOw">Introduction to Transformers - Part 1</a></li>
        <li><a target="_blank" href="https://www.youtube.com/watch?v=mMa2PmYJlCo">Introduction to Transformers - Part 2</a></li>
        <li><a target="_blank" href="https://www.youtube.com/watch?v=gJ9kaJsE78k&amp;t=2s">Introduction to Transformers - Part 3</a></li>
      </ul>
    </li>
    <li><a target="_blank" href="https://arxiv.org/abs/1706.03762">The paper that introduced transformers</a></li>
    <li><a target="_blank" href="https://arxiv.org/abs/2010.11929">The paper that introduced transformers for computer vision</a></li>
  </ul>
</div>
