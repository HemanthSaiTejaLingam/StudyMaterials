<div>
  <h2>Autoencoders</h2>
  <p><strong>Autoencoders</strong> are a very interesting neural network architecture that can be used for different applications directly (anomaly detection, denoising, ...), and it is also widely used in larger and modern architectures for other tasks (object detection, image segmentation).</p>
  <p>More advanced versions of autoencoders, such as variational autoencoders, can also be used as generative models, i.e., they can learn representations of data and use that representation to generate new realistic images.</p>
  <p>When studying CNNs for image classification or regression we have seen that the network is essentially composed of two parts: a backbone that extracts features from the image, and a Multi-Layer Perceptron or similar that uses those features to decide which class the image belongs to. In-between the two we have a flattening operation (or a Global Average Pooling layer) that takes the last feature maps coming out of the backbone and transforms them into a 1d array, which is a feature vector.</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/3e00929d-4299-4c42-9d2a-a670c26a0489'>
    <p>Structure of a typical CNN</pre>
  </pre>
  <p>Autoencoders have a similar <strong>backbone</strong> (called <strong>encoder</strong> in this context) that produces a <strong>feature vector</strong> (called <strong>embedding</strong> in this context). However, they substitute the <strong>fully-connected layers</strong> (the <strong>head</strong>) with a <strong>decoder</strong> stage whose scope is to reconstruct the input image starting from the embeddings:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/e60708fe-0c61-4a07-8a41-3193acf073fe'>
    <p>Typical structure of an autoencoder</pre>
  </pre>
  <p>This can appear pointless at first glance, but it is actually very useful in many contexts.</p>
  <h3>Uses of Autoencoders</h3>
  <ul>
    <li>Compress data</li>
    <li><strong>Denoise</strong> data</li>
    <li>Find outliers (do <strong>anomaly detection</strong>) in a dataset</li>
    <li><strong>Do inpainting</strong> (i.e., reconstruct missing areas of an image or a vector)</li>
    <li>With some modifications, we can use autoencoders as <strong>generative models</strong> - models capable of generating new images</li>
  </ul>
  <p>Autoencoders are also the basis for a whole field of research concerned with <strong>metric learning</strong>, which is learning representations of images that can be useful in downstream tasks.</p>
  <h3>Unsupervised vs. Supervised Learning</h3>
  <p>By looking closer at the structure and the tasks we have just described, you can see that autoencoders do not use the information on the label of the image at all. They are only concerned with the image itself, not with its label. The tasks that autoencoders address are examples of <i>unsupervised learning</i>, where the algorithm can learn from a dataset without any label. Another example of unsupervised learning that you might be familiar with is <i>clustering</i>.</p>
  <p>CNNs for image classification are instead an example of <i>supervised learning</i>, where the network learns to distinguish between classes by learning from a labeled dataset.</p>
  <h3>The Loss of Autoencoders</h3>
  <p>The autoencoder is concerned with encoding the input to a compressed representation, and then re-constructing the original image from the compressed representation. The signal to train the network comes from the differences between the input and the output of the autoencoder.</p>
  <p>For example, let's consider an autoencoder for images. We compare the input image to the output image and we want them to be as similar as possible.</p>
  <p>What is the right loss for this task?</p>
  <p>We have different possibilities, but the most common one is the Mean Squared Error (MSE) loss. It just considers the square of the difference between each pixel in the input image and the corresponding pixel in the output image, so minimizing this loss is equivalent to minimizing the difference of each pixel in the input with the corresponding pixel in the output. In practice, this is given by the formula:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/4f862345-3860-4384-9094-ab4d2e92f0da'>
  <p>We take the square of the difference between each pixel 𝑥_𝑖𝑗 in the input and the corresponding pixel 
𝑥^_𝑖𝑗 in the output and we average them out, i.e., we sum all the squared differences and then we divide by the number of pixels (which is equal to the number of rows 𝑛_rows times the number of columns 𝑛_cols).</p>
  <p>This loss is very simple and in practice gives good results. Using other types of losses is possible as well.</p>
</div>
<div>
  <h2>Autoencoders in Pytorch</h2>
  <pre><code>
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        ## encoder ##
        self.encoder = nn.Sequential(
            nn.Linear(28*28, encoding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(encoding_dim)
        )
        ## decoder ##
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 28*28),
            nn.Sigmoid()
        )
        self.auto_encoder = nn.Sequential(
            nn.Flatten(),
            self.encoder,
            self.decoder
        )
    def forward(self, x):
        # define feedforward behavior 
        # and scale the *output* layer with a sigmoid activation function
        encoded = self.auto_encoder(x)
        # Reshape the output as an image
        # remember that the shape should be (batch_size, channel_count, height, width)
        return encoded.reshape((x.shape[0], 1, 28, 28))</code>
  </pre>
  <p>We have built the simplest autoencoder, which is made up of two linear layers.We have trained it using the Mean Squared Error (MSE) loss. Of course, we did not use the labels, since anomaly detection with autoencoders is an unsupervised task.</p>
  <h3>Learnable Upsampling</h3>
  <p>We have seen how to use linear layers to create an autoencoder. Since we are working on images, it is natural to use convolution instead of just linear layers. Convolution allows us to keep spatial information and get a much more powerful representation of the content of an image.</p>
  <p>However, this poses a problem: while the encoder section can be just the backbone of a standard CNN, what about the decoder part? Yes, we could flatten the output of the backbone and then use linear layers to decode. But there are also other ways to upsample a compact representation into a full-resolution image. For example, we can use a Transposed Convolutional Layer, which can learn how to best upsample an image. </p>
  <h3>Transposed Convolutions</h3>
  <p>The Transposed Convolution can perform an upsampling of the input with learned weights. In particular, a Transposed Convolution with a 2 x 2 filter and a stride of 2 will double the size of the input image.</p>
  <p>Whereas a Max Pooling operation with a 2 x 2 window and a stride of 2 reduces the input size by half, a Transposed Convolution with a 2 x 2 filter and a stride of 2 will double the input size.</p>
  <p>Let's consider an autoencoder with two Max Pooling layers in the encoder, both having a 2 x 2 window and a stride of 2. If we want the network to output an image with the same size as the input, we need to counteract the two Max Pooling layers in the encoder with two Transposed Convolution layers with a 2 x 2 filter and a stride of 2 in the decoder. This will give us back an output with the same size as the input.</p>
  <h3>Transposed Convolutions in PyTorch</h3>
  <p>You can generate a Transposed Convolution Layer in PyTorch with:</p>
  <pre><code>unpool = nn.ConvTranspose2d(input_ch, output_ch, kernel_size, stride=2)</code></pre>
  <p>See the <a target='_blank' href='https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html'>full documentation on ConvTranspose2d</a> for more details.</p>
  <p>For example, we can generate a Transposed Convolution Layer that doubles the size of an input grayscale image and generates 16 feature maps as follows:</p>
  <pre><code>unpool = nn.ConvTranspose2d(1, 16, 2, stride=2)</code></pre>
  <h3>Alternative to a Transposed Convolution</h3>
  <p>The Transposed Convolutions tend to produce checkerboard artifacts in the output of the networks as <a target='_blank' href='https://distill.pub/2016/deconv-checkerboard/'>detailed in this Distill article</a>. Therefore, nowadays many practitioners replace them with a nearest-neighbor upsampling operation followed by a convolution operation. The convolution makes the image produced by the nearest-neighbors smoother. For example, we can replace this Transposed Convolution:</p>
  <pre><code>unpool = nn.ConvTranspose2d(1, 16, 2, stride=2)</code></pre>
  <p>with</p>
  <pre><code>
unpool = nn.Sequential(
    nn.Upsample(scale_factor = 2, mode='nearest'),
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
)</code>
  </pre>
  <p>The simplest autoencoder using CNNs can be constructed with a convolutional layer followed by Max Pooling, and then an unpooling operation (such as a Transposed Convolution) that brings the image back to its original size:</p>
  <pre>
    <code>
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        ## encoder ##
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        ## decoder ##
        self.decoder = nn.Sequential(
            # Undo the Max Pooling
            nn.ConvTranspose2d(3, 1, 2, stride=2),
            nn.Sigmoid()
        )
        self.auto_encoder = nn.Sequential(
            self.encoder,
            self.decoder
        )
    def forward(self, x):
        # define feedforward behavior 
        # and scale the *output* layer with a sigmoid activation function
        return self.auto_encoder(x)</code>
  </pre>
  <p>Of course, this autoencoder is not very performant. Typically you want to compress the information much more with a deeper encoder, and then uncompress it with a deeper decoder.</p>
  <p>In real-life situations, you can also use an already-existing architecture like a ResNet to extract the features (just remember to remove the final linear layers, i.e., the head and only keep the backbone). Of course, your decoder needs to then start from the embedding built by the architecture to get back to the dimension of the input image.</p>
  <h3>Denoising</h3>
  <p>We call <strong>denoising</strong> the task of removing noise from an image by reconstructing a denoised image</p>
  <p>This is a task that convolutional autoencoders are well-suited for.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/1e31e3da-0eb2-44f0-a509-071ab523f987'>
  <p>A denoising autoencoder is a normal autoencoder, but trained in a specific way.</p>
  <h3>How Do We Train a Denoising Autoencoder?</h3>
  <p>In order to train a denoising autoencoder we need to have access to the denoised version of the images. The easiest way to do this is to build a training dataset by taking clean images and adding noise to them. Then we will feed the image with the added noise into the autoencoder, and ask it to reconstruct the denoised (original) version.</p>
  <p>It is very important that we then compute the loss by comparing the input <i>uncorrupted image</i> (without noise) and the output of the network. DO NOT use the noisy version when computing the loss, otherwise your network will not learn!</p>
  <h3>Why Does it Work?</h3>
  <p>Let's consider an autoencoder trained on a noisy version of the MNIST dataset. During training, the autoencoder sees many examples of all the numbers. Each number has noisy pixels in different places. Hence, even though each number is corrupted by noise, the autoencoder can piece together a good representation for each number by learning different pieces from different examples. Here the convolutional structure helps a lot, because after a few layers the convolution smooths out a lot of the noise in a blurry but useful image of the number. This is also why generally you need to go quite deep with CNN autoencoders if you want to use them for denoising.</p>
  <p><strong>Variational autoencoder (VAE)</strong>: An extension of the idea of autoencoders that transforms them into proper generative models.</p>
</div>
