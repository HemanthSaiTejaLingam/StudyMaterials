<div>
  <h2>Augmentation Pipelines</h2>
  <p>A typical training augmentation pipeline is represented in this diagram.</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/6d676082-a2f3-4931-b7e7-1003458418a5'>
  <p>This typical training augmentation pipeline can be implemented in PyTorch as follows (transforms are documented in the <a href='https://pytorch.org/vision/main/transforms.html'>PyTorch transforms documentation</a>):</p>
  <pre>
    <code>
import torchvision.transforms as T
train_transforms = T.Compose(
    [
        # The size here depends on your application. Here let's use 256x256
        T.Resize(256),
        # Let's apply random affine transformations (rotation, translation, shear)
        # (don't overdo here!)
        T.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10),
        # Color modifications. Here I exaggerate to show the effect 
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # Apply an horizontal flip with 50% probability (i.e., if you pass
        # 100 images through around half of them will undergo the flipping)
        T.RandomHorizontalFlip(0.5),
        # Finally take a 224x224 random part of the image
        T.RandomCrop(224, padding_mode="reflect", pad_if_needed=True),  # -
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)</code>
  </pre>
  <p>This pipeline produces variations of the same image as this example:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/8750f2f2-650d-4e6b-a18d-f59d8eda119d'>
  <h3>Transformation Pipelines for Validation and Test</h3>
  <p>During validation and test you typically do not want to apply image augmentation (which is needed for training). Hence, this is a typical transform pipeline for validation and test that can be paired with the pipeline above:</p>
  <pre>
    <code>
testval_transforms = T.Compose(
    [
        # The size here depends on your application. Here let's use 256x256
        T.Resize(256),
        # Let's take the central 224x224 part of the image
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)</code>
  </pre>
  <p>Note that of course:</p>
  <ul>
    <li>The resize and crop should be the same as applied during training for best performance</li>
    <li>The normalization should be the same between training and inference (validation and test)</li>
  </ul>
  <h3>AutoAugment Transforms</h3>
  <p>There is a special class of transforms defined in <code>torchvision</code>, referred to as <a href='https://pytorch.org/vision/main/transforms.html#automatic-augmentation-transforms'>AutoAugment</a>. These classes implements augmentation policies that have been optimized in a data-driven way, by performing large-scale experiments on datasets such as ImageNet and testing many different recipes, to find the augmentation policy giving the best result. It is then proven that these policies provide good performances also on datasets different from what they were designed for.</p>
  <p>For example, one such auto-transform is called <code>RandAugment</code> and it is widely used. It is particularly interesting because it parametrizes the strength of the augmentations with one single parameter that can be varied to easily find the amount of augmentations that provides the best results. This is how to use it:</p>
  <pre><code>T.RandAugment(num_ops, magnitude)</code></pre>
  <p>The main parameters are:</p>
  <ul>
    <li><code>num_ops</code>: the number of random transformations applied. Defaut: 2</li>
    <li><code>magnitude</code>: the strength of the augmentations. The larger the value, the more diverse and extreme the augmentations will become.</li>
  </ul>
  <p>As usual, refer to the <a href='https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html#torchvision.transforms.RandAugment'>official documentation</a> for details.</p>
</div>
<div>
  <h2>BatchNorm for Convolutional Layers</h2>
  <p>BatchNorm can be used very easily in PyTorch as part of the convolutional block by adding the <code>nn.BatchNorm2d</code> layer just after the convolution:</p>
  <pre>
    <code>
self.conv1 = nn.Sequential(
  nn.Conv2d(3, 16, kernel_size=3, padding=1),
  nn.BatchNorm2d(16),
  nn.MaxPool2d(2, 2),
  nn.ReLU(),
  nn.Dropout2d(0.2)
)</code>
  </pre>
  <p>The only parameter is the number of input feature maps, which of course must be equal to the output channels of the convolutional layer immediately before it</p>
  <p>NOTE: It is important to use <code>BatchNorm</code> before DropOut. The latter drops some connections only at training time, so placing it before <code>BatchNorm</code> would cause the distribution seen by BatchNorm to be different between training and inference.</p>
  <h2>BatchNorm for Dense Layers</h2>
  <p>We can add BatchNorm to MLPs very easily by using <code>nn.BatchNorm1d</code>:</p>
  <pre>
    <code>
self.mlp = nn.Sequential(
  nn.Linear(1024, 500),
  nn.BatchNorm1d(500),
  nn.ReLU(),
  nn.Dropout(0.5)
)</code>
  </pre>
  <h2>Learning Rate Schedulers-StepLR</h2>
  <pre><code>
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# Training loop
for ... 
    ...
    # Update the weights
    optimizer.step()
    # Update the learning rate in the
    # optimizer according to the schedule
    scheduler.step()</code>
  </pre>
</div>
