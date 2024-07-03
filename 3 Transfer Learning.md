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
<div>
  <h2>State of the Art Models for Computer Vision</h2>
  <p>Vision Transformers have state-of-the-art performances in many academic computer vision tasks. CNNs are, however, still by far the most widely-used models for real-world computer vision applications.</p>
  <p>Transformers are very powerful but they need a lot more data than CNNs, and they are typically slower and more computationally expensive. CNNs are more data-efficient because they are built around two baked-in assumptions: local connectivity, which dictates that pixels close to each other are related (by using small kernels); and weight sharing, which dictates that different portions of an image must be processed identically (by sliding the same convolutional kernel across the entire image). Transformers are much more general, and they do not impose these assumptions. Therefore, they are more adaptable, but need more data to learn these characteristics of many images.</p>
  <p>There are also architectures that are hybrids of CNNs and Transformers, which try to create the best combination of both, aiming to be data-efficient but more powerful than pure CNNs.</p>
  <p>Summarizing, there are currently 3 categories of computer vision models:</p>
  <ul>
    <li>Pure CNN architectures - still widely used for the majority of real-world applications. Examples: <a href="https://arxiv.org/abs/2104.00298" target="_blank" rel="noopener noreferrer">EfficientNet V2</a>, <a href="https://arxiv.org/abs/2201.03545" target="_blank" rel="noopener noreferrer">ConvNeXt</a></li>
    <li>Pure Vision Transformers - currently widely used in academic environments and in large-scale real-world applications. Examples: <a href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener noreferrer">ViT</a>, <a href="https://arxiv.org/abs/2111.09883" target="_blank" rel="noopener noreferrer">Swin V2</a></li>
    <li>Hybrid architectures that mix elements of CNNs with elements of Transformers. Example: <a href="https://arxiv.org/abs/2106.04803" target="_blank" rel="noopener noreferrer">CoatNet</a></li>
  </ul>
  <p>As a final note, Transformers are now becoming even more important because they form the basis for multi-modal models - models that deal with, for example, image and text simultaneously. Examples of these are Open AI's <a href="https://openai.com/blog/clip/" target="_blank" rel="noopener noreferrer">CLIP</a> and Google's <a href="https://imagen.research.google/" target="_blank" rel="noopener noreferrer">ImageGen</a>.</p>
</div>
<div>
  <h2>Transfer learning</h2>
  <p><strong>Transfer learning</strong> is a technique that allows us to re-use what a network has learned on one dataset to learn about a different dataset.</p>

  <p>While training from scratch requires large datasets and a lot of resources, transfer learning can be applied successfully on much smaller datasets without the need for large computational resources.</p>
</div>
<div>
  <h2>Reusing Pre-Trained Networks</h2>
  <p>A normal CNN extracts more and more abstract features the deeper you go in the network. This means that the initial layers, which extract elementary features such as edges and colors, are probably pretty general and can be applied similarly on many different datasets. Instead, the last layers (especially the fully-connected layers) are highly specialized in the task they have been trained on.</p>
  <p>Accordingly, in transfer learning we keep the initial layers (that are pretty universal) unchanged or almost unchanged, while we change the last layers that must be specialized by task.</p>
  <p>How many layers we keep or modify slightly, and how many layers we change dramatically or even replace, depends on how similar our dataset is to the original dataset and on how much data we have.</p>
  <p>So essentially the transfer-learning workflow consists of taking a pre-trained model, freezing some of the initial layers and freeing or substituting some late layers, then training on our dataset.</p>
</div>
<div>
  <h2>Transfer Learning</h2>
  <p>Transfer learning involves taking a pre-trained neural network trained on a source dataset (for example, Imagenet) and adapting it to a new, different dataset, typically a custom dataset for a specific problem.</p>
  <p>There are different types of transfer learning and different strategies that you can use depending on:</p>
  <ol>
    <li>The size of your dataset</li>
    <li>How similar your dataset is to the original dataset that the network was trained on (e.g., ImageNet)</li>
  </ol>
  <p>We can schematize the different possibilities like this:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/4323689b-e2b6-4450-a794-42bc776dea8d'>
    <p>Description of the transfer learning regimes</pre>
  </pre>
  <h2>Dataset Size</h2>
  <p>It is difficult to define what a small dataset or a large dataset is exactly. However, for a typical classification example, a small dataset is in the range of 1000-10,000 images. A large dataset can have 100,000 images or more. These boundaries also change significantly depending on the size of the model you are using. A dataset that is large for a ResNet18 model (a ResNet with 18 layers) could be small for a ResNet150 architecture (a ResNet with 150 layers). The latter has many more parameters and a much larger capacity so it needs more data. In general, the larger the model, the more data it needs.</p>
  <h2>Dataset Similarity</h2>
  <p>Similarly, it is sometimes difficult to judge whether a target dataset is similar to the source dataset. For example, if the source dataset is Imagenet and the target dataset is of natural images, then the two datasets are pretty similar. However, if the target is medical images then the datasets are fairly dissimilar. However, it must be noted that CNNs look at images differently than we do, so sometimes datasets that look different to us are sufficiently similar for the model, and vice-versa. It is important to experiment and verify our assumptions.</p>
  <div>
  <h3>Size of Dataset: What to Do</h3>
  <h4>Small target dataset, similar to the source dataset: Train the head</h4>
  <p>This is a typical case, and the case where transfer learning really shines. We can use the pre-trained part of the network to extract meaningful feature vectors and use them to classify our images.</p>
  <p>In practice we take the head of the network and we substitute it with one or more new fully-connected layers (with the usual BatchNorm and ReLU layers in-between). Remember that the head of the network is the final part of the network, made typically by an MLP or similar, after the convolution part. It takes the output of the feature extractor part (also called <em>backbone</em>) and uses it to determine the class of the image (in the case of image classification). In some architectures like ResNet the head is just one layer (the last layer); in other architectures it is more complicated, encompassing the last few layers. Of course, the last of these layers needs to have as many output nodes as classes in our problem (or one number in case of regression).</p>
  <p>Then we train, keeping all the layers fixed except for the layer(s) we have just added.</p>
  <p>For example, let's say we have 1000 images (a small dataset) and a classification task with 10 classes. This is what we could do:</p>
  <pre>
    <code>
import torch
import torchvision.models as models
## Get a pre-trained model from torchvision, for example
## ResNet18
model = models.resnet18(pretrained=True)
## Let's freeze all the parameters in the pre-trained
## network
for param in model.parameters():
    param.requires_grad = False
## Through Netron.app we have discovered that the last layer is called
## "fc" (for "fully-connected"). Let's find out how many input features
## it has
input_features = model.fc.in_features
## We have 10 classes
n_classes = 10
## Let's substitute the existing fully-connected last layer with our
## own (this will have all its parameters free to vary)
model.fc = nn.Linear(input_features, n_classes)
## or we can use a more complicated head (this might or might not
## lead to improved performances depending on the case)
model.fc = nn.Sequential(
    nn.BatchNorm1d(input_features),
    nn.Linear(input_features, input_features * 2),
    nn.ReLU(),
    nn.BatchNorm1d(input_features * 2),
    nn.Dropout(0.5),
    nn.Linear(input_features * 2, n_classes)
)</code>
  </pre>
  <p>Now we can train our model as usual. You might want to start by executing the learning rate finder we have seen in our previous exercises and train for a few epochs. Depending on the size of your dataset, you might reach good performances rather quickly. It is likely that you will have to train for much less time than you would if you were to train from scratch. Be careful with overfitting and do not overtrain! If needed, also add more image augmentations, weight decay, and other regularization techniques.</p>
  <h4>Large dataset, at least somewhat similar to the original dataset: Fine-tune the entire network</h4>
  <p>If we have more data and/or the task is not very similar to the task that the network was originally trained to solve, then we are going to get better performances by fine-tuning the entire network.</p>
  <p>We start by performing the same procedure as the previous case: we remove the existing head, we freeze everything and we add our own head, then we train for a few epochs. Typically 1 or 2 epochs are sufficient.</p>
  <p>We then free all the layers and we train until convergence (until the validation loss stops decreasing). We need to be very careful to use a conservative learning rate, to avoid destroying what the network has learned during the original task. A good choice is typically a value between 2 and 10 times smaller than the learning rate we used to train the head. As usual, experimentation is typically needed to find the best learning rate for this phase.</p>
  <p>A more advanced technique that works well in practice is to use a <a target="_blank" rel="noopener noreferrer" href="https://arxiv.org/abs/1801.06146v5">different learning rate for every layer</a>. You start with using the maximum learning rate for the last layer and you gradually decrease the learning rate for layers deeper into the network until you reach the minimum for the first convolutional layer.</p>
  <h4>Large dataset, very different than the original dataset: Train from scratch</h4>
  <p>In this situation, fine-tuning does not give us better performance or faster training. We are better off just training from scratch. We can still take advantage of good architectures that performed well on ImageNet, since they are likely to work well on our dataset as well. We can just use them without pre-trained weights, for example:</p>
  <pre>
    <code>
import torch
import torchvision.models as models
## Get a pre-trained model from torchvision, for example
## ResNet18
model = models.resnet18(pretrained=False)
    </code>
  </pre>
  <h4>Small dataset, very different than the original dataset: Gather more data or use semi-supervised learning</h4>
  <p>This is the hardest situation. If you have tried fine-tuning just the head and it did not perform well enough, and fine-tuning more layers resulted in overfitting, you probably need to either collect more data or look into starting from scratch and use <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Semi-supervised_learning">semi-supervised learning</a>.</p>
  <h4>Other situations</h4>
  <p>When it is not clear whether you are in any of the situations described above, you can take approaches that are in-between.</p>
  <p>For example, when you have a dataset that is not very small but not very large either, you might get good performances by fine-tuning not only the head, but also a few of the last convolutional layers or blocks. Indeed, these layers encode high-level concepts such as "squares," "triangles," or textures, and therefore can improve by being fine-tuned or even trained from scratch on your data. Just free those layers along with the new head and train those, while keeping the rest fixed. Depending once again on the size of your data and the similarity with the original dataset, you can fine-tune these layers or reset them and train them from scratch. As usual, it takes a bit of experimentation to find the best solution.</p>
</div>
</div>
<div>
  <h2>TIMM: A Very Useful Library for Fine-Tuning</h2>
  <p>
    When performing fine-tuning (or training with a predefined architecture), we cannot avoid mentioning the fantastic
    <a href="https://github.com/rwightman/pytorch-image-models" target="_blank" rel="noopener noreferrer">PyTorch Image Models (timm) library</a>
    . It contains hundreds of models, many with pre-trained weights, and it keeps getting updated with the very latest architectures from the research community. It is very easy to use it for transfer learning. It is an alternative to
    <code>torchvision</code>
    that contains many more pretrained models.
  </p>
  <p>First let's install it with:</p>
  <pre>
    <code>pip install timm</code>
  </pre>
  <p>
    Then we can get a pre-trained model with a custom head just by doing:
  </p>
  <pre>
    <code>
n_classes = 196
model = timm.create_model("convnext_small", pretrained=True, num_classes=n_classes)</code>
  </pre>
  <p>
    The library already builds a head for you, so you do not need to build one explicitly.
  </p>
  <p>
    We can now choose to freeze some or all the layers except the last one, depending on the size of our dataset and its characteristics, and apply the techniques we discussed before.
  </p>
  <p>
    Note that you do not need to know the details of the architecture to be able to make a new head for it, as <code>timm</code> does that for you.
  </p>
</div>
