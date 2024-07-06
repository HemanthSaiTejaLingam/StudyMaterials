<div>
  <h2>Object Detection and Segmentation</h2>
  <p>When performing computer vision on images, we can ask a neural network to complete different types of tasks, with varying complexity:</p>
  <table>
  <tbody>
    <tr>
      <td>a</td>
      <td><strong>Image classification</strong></td>
      <td>Assign one or more labels to an image</td>
    </tr>
    <tr>
      <td>b</td>
      <td><strong>Object localization</strong></td>
      <td>Assign a label to most prominent object, define a box around that object</td>
    </tr>
    <tr>
      <td>c</td>
      <td><strong>Object detection</strong></td>
      <td>Assign a label and define a box for <em>all</em> objects in an image</td>
    </tr>
    <tr>
      <td>d</td>
      <td><strong>Semantic segmentation</strong></td>
      <td>Determine the class of each <em>pixel</em> in the image</td>
    </tr>
    <tr>
      <td>e</td>
      <td><strong>Instance segmentation</strong></td>
      <td>Determine the class of each <em>pixel</em> in the image distinguishing different instances of the same class</td>
    </tr>
  </tbody>
</table>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/78551c77-a351-49a3-8ebf-725e112f0a71'>
  <h2>Object Localization and Bounding Boxes</h2>
  <p><strong>Object localization</strong> is the task of assigning a label and determining the bounding box of an object of interest in an image.</p>
  <p>A <strong>bounding box</strong> is a rectangular box that completely encloses the object, whose sides are parallel to the sides of the image.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/ce33d273-9978-4842-ad63-a1f453ae6e4e'>
  <p>There are different ways of describing a bounding box, but they all require 4 numbers. These are some examples:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/2abcb2f9-ba2e-43c7-b037-c9dfb7add438'>
  <p>Note in these images above that x_min, y_min is in the upper left. Images in deep learning are usually indexed as matrices, starting from the upper left. So if an image is 100 x 100, then (0,0) is the point in the upper left, and (99,99) is the point in the lower right.</p>
  <h3>Architecture of Object Localization Networks</h3>
  <p>The architecture of an object localization network is similar to the architecture of a classification network, but we add one more head (the localization head) on top of the existing classification head:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/9f573794-6a9b-46a0-b655-a9da54b09797'>
  <h3>A Multi-Head Model in PyTorch</h3>
  <p>A <strong>multi-head model</strong> is a CNN where we have a backbone as typical for CNNs, but more than one head. For example, for object localization this could look like:</p>
  <pre>
    <code>
from torch import nn
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: this can be a custom network, or a
        # pre-trained network such as a resnet50 where the
        # original classification head has been removed. It computes
        # an embedding for the input image
        self.backbone = nn.Sequential(..., nn.Flatten())
        # Classification head: an MLP or some other neural network
        # ending with a fully-connected layer with an output vector
        # of size n_classes
        self.class_head = nn.Sequential(..., nn.Linear(out_feature, n_classes))
        # Localization head: an MLP or some other neural network
        # ending with a fully-connected layer with an output vector
        # of size 4 (the numbers defining the bounding box)
        self.loc_head = nn.Sequential(..., nn.Linear(out_feature, 4))
    def forward(self, x):
        x = self.backbone(x)
        class_scores = self.class_head(x)
        bounding_box = self.loc_head(x)
        return class_scores, bounding_box</code>
  </pre>
  <h3>Loss in a Multi-Head Model</h3>
  <p>An object localization network, with its two heads, is an example of a multi-head network. Multi-head models provide multiple outputs that are in general paired with multiple inputs.</p>
  <p>For example, in the case of object localization, we have 3 inputs as ground truth: the image, the label of the object ("car") and the bounding box for that object (4 numbers defining a bounding box). The object localization network processes the image through the backbone, then the two heads provide two outputs:</p>
  <ol>
    <li>The class scores, that need to be compared with the input label</li>
    <li>The predicted bounding box, that needs to be compared with the input bounding box.</li>
  </ol>
  <p>The comparison between the input and predicted labels is made using the usual Cross-entropy loss, while the comparison between the input and the predicted bounding boxes is made using, for example, the mean squared error loss. The two losses are then summed to provide the total loss L.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/f92e1240-e90c-49bb-b877-dc68b0c28c68'>
  <p>Since the two losses might be on different scales, we typically add a hyperparameter α to rescale one of the two:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c7398027-62a7-495b-a160-d0cd3e508209'>
  <h3>Multiple Losses in PyTorch</h3>
  <p>This is an example of a training loop for a multi-head model with multiple losses (in this case, cross-entropy and mean squared error):</p>
  <pre>
    <code>
class_loss = nn.CrossEntropyLoss()
loc_loss = nn.MSELoss()
alpha = 0.5
...
for images, labels in train_data_loader:
    ...
    # Get predictions
    class_scores, bounding_box = model(images)
    # Compute sum of the losses
    loss = class_loss(class_scores) + alpha * loc_loss(bounding_box)
    # Backpropagation
    loss.backward()
    optimizer.step()</code>
  </pre>
  <h2>Object Detection</h2>
  <p>The task of <strong>object detection</strong> consists of detecting and localizing all the instances of the objects of interest.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/adda11f3-4418-42cf-af9c-ef1fe289acc0'>
  <p>Different images of course can have a different number of objects of interest: for example an image could have one car, zero cars, or many cars. The same images could also have one person, zero people or many people. In each one of these cases, we need the network to output one vector of class scores plus 4 numbers to define the bounding box for each object. How do we build a network with a variable number of outputs?</p>
  <p>It is clear that a network with the same structure as the object localization network would not work, because we would need a variable number of heads depending on the content of the image.</p>
  <p>One way would be to slide a window over the image, from the upper left corner to the lower right corner, and for each location of the image we run a normal object localization network. This <strong>sliding window</strong> approach works to a certain extent, but is not optimal because different objects can have different sizes and aspect ratios. Thus, a window with a fixed size won't fit well all objects of all sizes. For example, let's consider cars: depending on how close or far they are, their size in the image will be different. Also, depending on whether we are seeing the back or the front, or the side of the car, the aspect ratio of the box bounding the object would be pretty different. This becomes even more extreme if we consider objects from different classes, for example cars and people:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c08a32f7-351d-4e2b-8bb6-280d56a48596'>
    <p>Objects in images have different scales and aspect ratios, so a window that is optimal for an object type at a certain scale won't be optimal for a different scale or a different object class</p>
  </pre>
  <p>Nowadays there are two approaches to solving the problem of handling a variable number of objects, and of their different aspect ratios and scales:</p>
  <h3>1) One-stage object detection</h3>
  <p>We consider a fixed number of windows with different scales and aspect ratios, centered at fixed locations (<strong>anchors</strong>). The output of the network then has a fixed size. The localization head will output a vector with a size of 4 times the number of anchors, while the classification head will output a vector with a size equal to the number of anchors multiplied by the number of classes.</p>
  <h3>2) Two-stage object detection</h3>
  <p>In the first stage, an algorithm or a neural network proposes a fixed number of windows in the image. These are the places in the image with the highest likelihood of containing objects. Then, the second stage considers those and applies object localization, returning for each place the class scores and the bounding box.</p>
  <p>In practice, the difference between the two is that while the first type has fixed anchors (fixed windows in fixed places), the second one optimizes the windows based on the content of the image.</p>
  <h5>Resources for Two-Stage Object Detection</h5>
  <a target='_blank' href='https://www.youtube.com/watch?v=6I3m0SsLPo4'>Video: Two-stage object detection</a>
  <a target="_blank" href="https://iopscience.iop.org/article/10.1088/1742-6596/1544/1/012033/meta">object detection algorithms</a>
  <h2>One-Stage Object Detection: RetinaNet</h2>
  <p>The RetinaNet network is an example of a one-stage object detection algorithm. Like many similar algorithms, it uses <strong>anchors</strong> to detect objects at different locations in the image, with different scales and aspect ratios.</p>
  <p><strong>Anchors</strong> are windows with different sizes and different aspect ratios, placed in the center of cells defined by a grid on the image:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/bb3d2a5f-d5dc-49f3-b5b3-a3eb8a28472d'>
    <p>Grid (in green) and windows (in red)</p>
  </pre>
  <p>We divide the image with a regular grid. Then for each grid cell we consider a certain number of windows with different aspect ratios and different sizes. We then "anchor" the windows in the center of each cell. If we have 4 windows and 45 cells, then we have 180 anchors.</p>
  <p>We run a localization network considering the content of each anchor. If the class scores for a particular class are high for an anchor, then we consider that object detected for that anchor, and we take the bounding box returned by the network as the localization of that object.</p>
  <p>This tends to return duplicated objects, so we post-process the output with an algorithm like <a target='_blank' href='https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/'>Non Maximum Suppression</a>.</p>
  <h2>Feature Pyramid Networks (FPNs)</h2>
  <p>RetinaNet uses a special backbone called Feature Pyramid Network.</p>
  <p>The <a target='_blank' href='https://arxiv.org/abs/1612.03144'>Feature Pyramid Network</a> is an architecture that extracts multi-level, semantically-rich feature maps from an image:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/32d0d8db-e3a1-43bd-99b6-354861b27010'>
  <p>A regular CNN backbone of convolution, pooling, and other typical CNN layers is used to extract multiple feature maps from the original image (downsampling path). The last feature map contains the most semantic-rich representation, which is also the least detailed because of the multiple pooling. So, we copy that to the upsampling path and run object detection with anchors on that. Then, we upsample it and sum it to the feature map from the same level in the downsampling path. This means we are mixing the high-level, abstract information from the feature map in the upsampling path to the more detailed view from the downsampling path. We then run object detection on the result. We repeat this operation several times (3 times in total in this diagram). This is how RetinaNet uses the Feature Pyramid Network:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/fe4bb678-443b-4dd6-a723-7090cb74b15b'>
    <p>The RetinaNet architecture</p>
  </pre>
  <p>RetinaNet conducts object classification and localization by independently employing anchors at each Feature Pyramid level. The diagram above illustrates that the classification subnet and the box regression subnet pass through four convolutional layers with 256 filters. Subsequently, they undergo convolutional layers with KA and 4A filters for the classification and localization, respectively.</p>
  <h2>Focal Loss</h2>
  <p>The third innovative feature of RetinaNet is the so-called <strong>Focal Loss</strong>.</p>
  <p>When using a lot of anchors on multiple feature maps, RetinaNet encounters a significant class balance problem: most of the tens of thousands of anchors used in a typical RetinaNet will not contain objects. The crop of the image corresponding to these anchors will normally be pretty easy to classify as background. So the network will very quickly become fairly confident on the background. The normal cross-entropy loss assigns a low but non-negligible loss even to well-classified examples. For example, let's look at the blue curve here:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/ff923c03-094c-4e04-bb38-f557d18d124b'>
    <p>Comparison of loss functions: cross-entropy vs. focal loss across different confidence levels</p>
  </pre>
  <p>Here we are considering for simplicity a binary classification problem. Let's consider an example having a positive label. If our network has a confidence on the positive label of 0.8, it means it is classifying this example pretty well: it is assigning the right label, with a good confidence of 0.8. However, the loss for this example is still around 0.5. Let's now consider a different positive example, where the network is assigning a probability for the positive class of 0.1. This is a False Negative, and the loss accordingly is pretty high (around 4). Let's now assume that we have 10 examples where the network is correct and has a confidence of 0.8 (and loss of 0.5), and one example where the network is wrong and has a loss of 4. The ten examples will have a cumulative loss of 0.5 x 10 = 5, which is larger than 4. In other words, the cumulative loss of the examples that are already classified well is going to dominate over the loss of the example that is classified wrong. This means that the backpropagation will try to make the network more confident on the 10 examples it is already classifying well, instead of trying to fix the one example where the network is wrong. This has catastrophic consequences for networks like RetinaNet, where there are usually tens of thousands of easy background anchors for each anchor containing an object.</p>
  <p>The <strong>Focal Loss</strong> adds a factor in front of the normal cross-entropy loss to dampen the loss due to examples that are already well-classified so that they do not dominate. This factor introduces a hyperparameter γ: the larger γ, the more the loss of well-classified examples is suppressed.</p>
  <p>RetinaNet is characterized by three key features:
  1)Anchors<br>
  2)Feature Pyramid Networks<br>
  3)Focal loss<br>
  </p>
  <h2>Intersection over Union (IoU)</h2>
  <p>The <strong>IoU</strong> is a measure of how much two boxes (or other polygons) coincide. As the name suggests, it is the ratio between the area of the intersection, or overlap, and the area of the union of the two boxes or polygons:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c6c24d75-df29-4a47-bbb1-db49e355b14a'>
  <p>IoU is a fundamental concept useful in many domains, and is a key metric for the evaluation of object detection algorithms.</p>
  <h3>Mean Average Precision (mAP)</h3>
  <p><strong>Mean Average Precision (mAP)</strong> conveys a measurement of precision averaged over the different object classes.</p>
  <p>Let’s say we have a number of classes. We consider all the binary classification problems obtained by considering each class in turn as positive and all the others as negative.</p>
  <p>For each one of these binary sub-problems, we start by drawing the precision-recall curve that is obtained by measuring precision and recall for different confidence level thresholds, while keeping the IoU threshold fixed (for example at 0.5). The confidence level is the classification confidence level, i.e., the maximum of the softmax probabilities coming out of the classification head. For example, we set the confidence threshold to 0.9 and measure precision and recall, then we change the threshold to say 0.89 and measure precision and recall, and so on, until we get to a threshold of 0.1. This constitutes our precision-recall curve:</p>
<pre>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/a2d4d640-249b-480e-aa21-b6528d11e3bd'>
  <p>Precision vs. recall with a varying confidence threshold</p>
</pre>
<p>We then interpolate the precision and recall curve we just obtained by using a monotonically-decreasing interpolation curve, and we take the area under the curve. This represents the so-called Average Precision (AP) for this class:</p>
<pre>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/d45491e6-291d-4178-bb95-0a9c786204b9'>
  <p>Precision-recall curve: original vs. interpolated</p>
</pre>
  <p>We repeat this procedure for all classes, then we take the average of the different APs and call that mean Average Precision, or mAP.</p>
  <p>A related metric is called mean Average Recall (mAR). Similarly to the mAP, we split our problem into a number of binary classification problems. For each class, we compute the recall curve obtained by varying this time the IoU threshold from 0.5 to 1. We can now consider the integral of the curve. Since we integrate between 0.5 and 1, and Recall is a quantity bounded between 0 and 1, the integral would be bounded between 0 and 0.5. We therefore multiply by 2 to make it a quantity bounded between 0 and 1. Twice the area under the recall curve represents the so-called Average Recall (AR) for this class:</p>
  <pre>
    <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/2ddc4c2a-e673-4eec-a8f0-1d42074292cc'>
    <p>Recall-IoU curve: Average Recall calculation</p>
  </pre>
  <p>We then take the average of the AR over the different classes, to define the mean Average Recall, or mAR.</p>
  <h3>Semantic Segmentation: UNet</h3>
  <p>The UNet is a specific architecture for semantic segmentation. It has the structure of a standard autoencoder, with an encoder that takes the input image and encodes it through a series of convolutional and pooling layers into a low-dimensional representation.</p>
  <p>Then the decoder architecture starts from the same representation and constructs the output mask by using transposed convolutions. However, the UNet adds skip connections between the feature maps at the same level in the encoder and in the decoder, as shown below:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/85cfaed5-e46f-4021-abc5-e78f2f27abc6'>
  <p>In the decoder, the feature map coming from the decoder path is concatenated along the channel dimension with the feature map coming from the encoder path. This means that the next transposed convolution layer has access to information with high semantic content coming from the previous decoder layer, along with information with high detail coming from the encoder path. The final segmentation mask is then a lot more detailed than what you would obtain with a simple encoder-decoder architecture without skip connections between the encoder and the decoder.</p>
  <h3>The Dice Loss</h3>
  <p>There are a few different losses that one can use for semantic segmentation. One loss that tend to work well in practice is called Dice loss, named after Lee Raymond Dice, who published it in 1945. Here is how Dice loss is calculated:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/80605b84-5229-4a5e-953c-b645ccaed86c'>
  <p>pi and yi represent the i-th pixel in respectively the prediction mask and the ground truth mask. The sums are taken over all the n_pix pixels in the image.</p>
  <p>The Dice loss derives from the F1 score, which is the geometric mean of precision and recall. Consequently, the Dice loss tends to balance precision and recall at the pixel level.</p>
  <h2>UNet in PyTorch</h2>
  <p>We will use the implementation of UNet provided by the wonderful open-source library <a target='_blank' href='https://github.com/chsasank/segmentation_models.pytorch'>segmentation_models for PyTorch</a>. The library also implements the Dice loss.</p>
  <p>This is how you can define a UNet using this library:</p>
  <pre>
    <code>
import segmentation_models_pytorch as smp
# Binary segmentation?
binary = True
n_classes = 1
model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        # +1 is for the background
        classes=n_classes if binary else n_classes + 1)</code>
  </pre>
  <p>The Dice loss is simply:</p>
  <pre><code>loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)</code></pre>
</div>
