<div>
  <h2>Loading Images in PyTorch</h2>
  <h3>Transforms</h3>
  <p>Before we can feed images to a neural network, there are some operations we must perform. As we have seen before, we need to normalize the images so the values in the pixels are floating point numbers between 0 and 1, or -1 and 1, instead of values from 0 to 255 as in the RGB format. We also need to transform them to PyTorch Tensors, so our network can handle them.In many cases we also want to augment our dataset by performing random transformations.</p>
  <div>
    <p>Let's see how to normalize the images and transform them to tensors:</p>
    <pre>
    <code>
      import torchvision.transforms as T
      # T.Compose creates a pipeline where the provided
      # transformations are run in sequence
      transforms = T.Compose([
      # This transforms takes a np.array or a PIL image of integers
      # in the range 0-255 and transforms it to a float tensor in the
      # range 0.0 - 1.0
      T.ToTensor(),
      # This then renormalizes the tensor to be between -1.0 and 1.0,
      # which is a better range for modern activation functions like
      # Relu
      T.Normalize((0.5), (0.5))])
    </code>
    </pre>
    <p>In many real-world datasets we also need some other operations like resizing.</p>
  </div>
</div>
<div>
  <h2>Datasets</h2>
  <p>PyTorch offers dataset and dataloaders specific for images and their annotations</p>
  <p>A <code>Dataset</code> instance provides methods to load and manipulate the data, whereas a <code>DataLoader</code> instance wraps the dataset and allows iterations over it during training, validation, and test.</p>
  <p>
    It is possible to define custom datasets when needed, but the
    <code>torchvision</code> library offers specific classes inheriting from the base
    <code>Dataset</code> class for all major computer vision datasets. You can find a list of these datasets on the
    <a target="_blank" href="https://pytorch.org/vision/stable/datasets.html#image-classification">PyTorch Vision Image Classification Datasets page</a>.
    For example, you can load the MNIST dataset just by doing:
  </p>
  <pre>
  <code>
        train_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms
        )
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms
        )
  </code>
  </pre>
  <p>
    <code>torchvision</code> also offers an <code>ImageFolder</code> class that can be used to extract images and labels directly from a local directory.
  </p> 
  <p>You have to structure your data in the following way:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/f9b189cc-722a-4ad9-a31c-3b165137b4b8'>
  <div>
      <p>We need a top-level directory with the name of your dataset (<code>custom_data</code> in this case). Then, we need one or more subdirectories representing the classes. In this case we have a dataset of cats and dogs, and accordingly we have two subdirectories with the same name. Within each subdirectory we place the images belonging to that class.</p>
      <p>The <code>ImageFolder</code> dataset class can auto-probe the classes (by using the names of the subdirectories) and can assign each image to the right class by looking into which subdirectory the image is placed.</p>
      <p>This is how to use the <code>ImageFolder</code> class:</p>
      <pre>
          <code>
              <span style="color: rgb(15, 43, 61); font-weight: bold;">from</span> torchvision <span style="color: rgb(15, 43, 61); font-weight: bold;">import</span> datasets
              train_data <span style="color: rgb(15, 43, 61); background: rgb(255, 255, 255);">=</span> datasets.<span style="color: rgb(15, 43, 61);">ImageFolder</span>(<span style="color: rgb(221, 17, 68);">"/data/custom_data"</span>, transform<span style="color: rgb(15, 43, 61); background: rgb(255, 255, 255);">=</span>transforms)
          </code>
      </pre>
  </div>
</div>
<div>
  <h3>Dataloaders</h3>
    <p>A dataloader allows sequential or random iterations over a dataset or over a subset of a dataset.</p>
    <p>For example, a dataloader for MNIST can be obtained as follows:</p>
    <pre>
        <code>train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)</code>
    </pre>
    <p>The parameter <code>batch_size</code> indicates the size of the mini-batch for Stochastic Gradient Descent, while <code>num_workers</code> indicates the number of processes that PyTorch should use to load the data. It is important to be able to feed the GPU with enough data to keep it busy, otherwise the training will be slow. By using several processes that are reading data in parallel, PyTorch can increase the GPU usage and the speed of training. A good rule of thumb is to use a number of workers equal to the number of CPUs on the current machine:</p>
    <pre>
      <code>
        import multiprocessing
        n_workers = multiprocessing.cpu_count()
      </code>
    </pre>
    <p>Once you have a dataloader, you can easily loop over all the data one batch at a time:</p>
    <pre>
        <code>
          for image_batch, label_batch in train_loader:
              ... do something ...
        </code>
    </pre>
    <p>If all you want to do is obtain the next batch, then:</p>
    <pre>
      <code>
        ## Get an iterator from the dataloader
        dataiter = iter(train_loader)
        ## Get the next batch
        image_batch, label_batch = dataiter.next()
      </code>
    </pre>
    <h4>Splitting Train and Validation Data</h4>
    <p>It is typical to extract a validation set from the training data, for things like hyperparameter optimization and to prevent overfitting by monitoring the relationship between train and validation loss. We also reserve a test set for testing after the model development has finished.</p>
    <p>It is easy to split a dataset using PyTorch:</p>
    <pre>
        <code>
          train_size = int(0.8 * len(train_data))
          val_size = len(train_data) - train_size
          train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
          train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
          val_loader = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        </code>
    </pre>
    <p>Here, <code>train_size</code> and <code>val_size</code> define the sizes of the training and validation sets respectively, and <code>random_split</code> splits the dataset randomly.</p>
</div>
<div>
      <h2>Recap of the Structure of a Neural Network in PyTorch</h2>
      <p>A model in PyTorch is implemented as a class with at least two methods: the <code>__init__</code> method and the <code>forward</code> method.</p>
      <p>The <code>__init__</code> method should initialize all the layers that the model uses, while the <code>forward</code> method implements a forward pass through the network (i.e., it applies all the layers to the input). Note that the backward pass for backpropagation is executed by PyTorch <a target="_blank" rel="noopener noreferrer" href="https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html"></a> and does not need to be implemented.</p>
      <p>So a typical model in PyTorch looks like this:</p>
      <pre>
        <code>
import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create layers. In this case just a standard MLP
        self.fc1 = nn.Linear(20, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)
    def forward(self, x):
        # Call the layers in the appropriate sequence
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
        </code>
      </pre>
      <p>Remember that at any time you can call your model like this:</p>
      <pre>
        <code>
# Make up some data
x = torch.rand(20)
m = MyModel()
out = m(x)
        </code>
      </pre>
      <p>This is useful when developing your own architecture because you can verify that the model runs (for example, you got all the shapes right) and also you can check the shape of the output.</p>
      <h3>Using nn.Sequential</h3>
      <p>When the network is just a simple sequential application of the different layers, you can use <code>nn.Sequential</code>, which allows saving a lot of boilerplate code. For example, the previous network can be written as:</p>
      <pre>
        <code>
import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create layers. In this case just a standard MLP
        self.model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
    def forward(self, x):
        # nn.Sequential will call the layers 
        # in the order they have been inserted
        return self.model(x)
        </code>
      </pre>
      <p>In fact, when creating a network like this, we can skip the definition of a custom class and just use <code>nn.Sequential</code> like this:</p>
      <pre>
        <code>
model = nn.Sequential(
    # Add layers here
)
        </code>
      </pre>
      <p>The first method (defining a custom class) is more flexible and allows the use of architectures that are not strictly sequential. Therefore, we will use it throughout this class. However, the second abbreviated form is useful in many real-world circumstances.</p>
</div>
<div>
  <h2>Design of an MLP - Rules of Thumb</h2>
            <p>When designing an MLP you have a lot of different possibilities, and it is sometimes hard to know where to start. Unfortunately there are no strict rules, and experimentation is key. However, here are some guidelines to help you get started with an initial architecture that makes sense, from which you can start experimenting.</p>
            <p>The number of inputs <code>input_dim</code> is fixed (in the case of MNIST images for example it is 28 x 28 = 784), so the first layer must be a fully-connected layer (<code>Linear</code> in PyTorch) with <code>input_dim</code> as input dimension.</p>
            <p>Also the number of outputs is fixed (it is determined by the desired outputs). For a classification problem it is the number of classes <code>n_classes</code>, and for a regression problem it is 1 (or the number of continuous values to predict). So the output layer is a <code>Linear</code> layer with <code>n_classes</code> (in case of classification).</p>
            <p>What remains to be decided is the number of hidden layers and their size. Typically you want to start from only one hidden layer, with a number of neurons between the input and the output dimension. Sometimes adding a second hidden layer helps, and in rare cases you might need to add more than one. But one is a good starting point.</p>
            <p>As for the number of neurons in the hidden layers, a decent starting point is usually the mean between the input and the output dimension. Then you can start experimenting with increasing or decreasing, and observe the performances you get. If you see <a href="https://en.wikipedia.org/wiki/Overfitting" target="_blank" >overfitting</a>, start by adding regularization (<a href="https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html" target="_blank" >dropout</a> and weight decay) instead of decreasing the number of neurons, and see if that fixes it. A larger network with a bit of drop-out learns multiple ways to arrive to the right answer, so it is more robust than a smaller network without dropout. If this doesn't address the overfitting, then decrease the number of neurons.
            If you see <a href="https://en.wikipedia.org/wiki/Overfitting" target="_blank">underfitting</a>, add more neurons. You can start by approximating up to the closest power of 2. Keep in mind that the number of neurons also depends on the size of your training dataset: a larger network is more powerful but it needs more data to avoid overfitting.</p>
            <p>So let's consider the MNIST classification problem. We have <code>n_classes = 10</code> and <code>input_dim = 784</code>, so a starting point for our experimentation could be:</p>
            <pre><code>
import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(784, 400),
            nn.Dropout(0.5),  # Dropout layer for regularization
            nn.ReLU(),  # Activation function
            # Hidden layer
            nn.Linear(400, 400),
            nn.Dropout(0.5),  # Dropout layer for regularization
            nn.ReLU(),  # Activation function
            # Output layer
            nn.Linear(400, 10)  # 10 classes for MNIST
        )
    def forward(self, x):
        # Forward pass
        return self.model(x)
</code></pre>
</div>
<div>
  <h2>Cross-Entropy Loss</h2>
  <p>The cross-entropy loss is the typical loss used for classification problems. It can be instanced in PyTorch like this:</p>
  <pre>
    <code>
from torch import nn
loss = nn.CrossEntropyLoss()
    </code>
  </pre>
  <p>
    In the <a href="https://pytorch.org/docs/stable/" target="_blank">PyTorch documentation<span>(opens in a new tab)</span></a>,
    you can see that the cross-entropy loss function actually involves two steps:
</p>
<ul>
  <li>It first applies a softmax function to any output it sees</li>
  <li>Then it applies <a href="https://pytorch.org/docs/stable/nn.html#nllloss" target="_blank" >NLLLoss</a>; negative log likelihood loss</li>
</ul>
<p>Then it returns the average loss over a batch of data.</p>
<p>Since the <code>nn.CrossEntropyLoss</code> already applies the softmax function, the output of our network should be unnormalized class scores, and NOT probabilities. In other words, we must NOT apply softmax in the <code>forward</code> method of our network.</p>
<h2>Another Approach</h2>
<p>We could also separate the softmax and NLLLoss steps.</p>
<ul>
  <li>
    In the <code>forward</code> function of our model, we would <em>explicitly</em> apply a softmax activation function (actually the logarithm of the softmax function, which is more numerically stable) to the output, <code>x</code>.
  </li>
<pre>
  <code>
import torch.nn.functional as F
 ...
 ...
    def forward(self, x):<br>
        ...<br>
        # a softmax layer to convert 10 outputs 
        # into a distribution of class probabilities
        return F.log_softmax(x, dim=1)
  </code>
</pre>
<li>Then, when defining our loss criterion, we would apply nn.NLLLoss instead of nn.CrossEntropyLoss.</li>
</ul>
<pre>
  <code>
    criterion = nn.NLLLoss()
  </code>
</pre>
<p>
  This separates the usual <code>loss = nn.CrossEntropyLoss()</code> into two steps. It is a useful approach should you want the output of a model to be class <em>probabilities</em> rather than class scores.
</p>
<p>
  Typically the first approach (using the Cross Entropy Loss) is preferred during training and validation (many tools actually assume that this is the case). However, when you export your model you might want to add the softmax at the end of your <code>forward</code> method, so that at inference time the output of your model will be probabilities and not class scores.
</p>
<h2>The Optimizer</h2>
<p>An optimizer is a class or a function that takes a function with parameters (typically our loss) and optimizes it. In the case of neural networks, optimization means minimization; i.e., the optimizer determines the values of the parameters that minimize the loss function. The problem indeed is formulated so that the parameters providing the minimum loss also provide the best performances.</p>
<p>PyTorch provides many optimizers. Two common ones are vanilla Stochastic Gradient Descent (SGD) and Adam. While the former is standard Gradient Descent applied using mini-batches, the latter is a more sophisticated algorithm that often provides similar results to SGD but faster. Both of them take as parameters the learning rate <code>lr</code> and (optionally) the regularization parameter <code>weight_decay</code>.</p>
<p>This is how to create optimizer instances in PyTorch:</p>
<pre>
  <code>
import torch.optim
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0)<br>
import torch.optim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
</code>
</pre>
<p>
  For other options as well as other available optimizers, please see the
  <a target="_blank" href="https://pytorch.org/docs/stable/optim.html">official documentation<span>(opens in a new tab)</span></a>.
</p>
<h2>The Training Loop</h2>
<pre>
  <code>
# number of epochs to train the model
n_epochs = 50
# Set model to training mode
# (this changes the behavior of some layers, like Dropout)
model.train()
# Loop over the epochs
for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    # Loop over all the dataset using the training
    # dataloader
    for data, target in train_dataloader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: 
        # compute predictions
        output = model(data)
        # calculate the loss which compare the model
        # output for the current batch with the relative
        # ground truth (the target)
        loss = criterion(output, target)
        # backward pass: 
        # compute gradient of the loss with respect to 
        # model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
  </code>
</pre>
<h2>Validation Set: Takeaways</h2>
<p>We create a validation set to:</p>
<ol role="list">
  <li>Measure how well a model generalizes, during training</li>
  <li>Tell us when to stop training a model; when the validation loss stops decreasing (and especially when the validation loss starts increasing and the training loss is still decreasing) we should stop training. It is actually more practical to train for a longer time than we should, but save the weights of the model at the minimum of the validation set, and then just throw away the epochs after the validation loss minimum.</li>
</ol>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/954f7c0c-61b3-4bd3-b05d-bbad9f0b5411'>
</div>
<div>
  <h2>Validation Loop</h2>
  <p>Once we have performed an epoch of training we can evaluate the model against the validation set to see how it is doing. This is accomplished with the validation loop:
</p>
  <pre>
    <code>
# Tell pytorch to stop computing gradients for the moment
# by using the torch.no_grad() context manager
with torch.no_grad():
  # set the model to evaluation mode
  # This changes the behavior of some layers like
  # Dropout with respect to their behavior during
  # training
  model.eval()
  # Keep track of the validation loss
  valid_loss = 0.0
  # Loop over the batches of validation data
  # (here we have removed the progress bar display that is
  # accomplished using tqdm in the video, for clarity)
  for batch_idx, (data, target) in enumerate(valid_dataloader):
    # 1. forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # 2. calculate the loss
    loss_value = criterion(output, target)
    # Calculate average validation loss
    valid_loss = valid_loss + (
      (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
    )
    # Print the losses 
  print(f"Epoch {epoch+1}: training loss {train_loss:.5f}, valid loss {valid_loss:.5f}")
    </code>
  </pre>
  <p>It is usually a good idea to wrap the validation loop in a function so you can return the validation loss for each epoch, and you can check whether the current epoch has the lowest loss so far. In that case, you save the weights of the model.</p>
<h2>The Test Loop</h2>
<p>The test loop is identical to the validation loop, but we of course iterate over the test dataloader instead of the validation dataloader.</p>
</div>
<div>
  <h2>Typical Image Classification Steps</h2>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/b6504387-d39b-417e-87f0-6c46f8bbe640'>
</div>
