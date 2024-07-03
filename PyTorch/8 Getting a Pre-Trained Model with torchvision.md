<div>
  <h2>Getting a Pre-Trained Model with <code>torchvision</code></h2>
  <p>You can find the list of all models supported by <code>torchvision</code> in the <a href="https://pytorch.org/vision/stable/models.html" target="_blank" >official documentation</a> (note that new models are added with each new version, so check that the list you are consulting is appropriate for the version of PyTorch you are using). Then you can load models by name. For example, let's load a <code>resnet</code> architecture:</p>
  <pre><code>
import torchvision.models
model = torchvision.models.resnet18(pretrained=True)</code>
  </pre>
  <p>The <code>pretrained=True</code> option indicates that we want the weights obtained after training on ImageNet or some other dataset. If we set <code>pretrained=False</code> we get the model initialized with the default initialization, ready to be trained from scratch.</p>
  <h3>Freezing and Thawing Layers and Parameters</h3>
  <p>A frozen parameter is a parameter that is not allowed to vary during training. In other words, backpropagation will ignore that parameter and won't change its value nor compute the gradient of the loss with respect to that parameter.</p>
  <p>In PyTorch you can freeze all the parameters of a network using the following code:</p>
  <pre><code>
for param in model.parameters():
    param.requires_grad = False</code>
  </pre>
  <p>Similarly, you can also freeze the parameters of a single layer. For example, say that this layer is called <code>fc</code>, then:</p>
  <pre>
    <code>
for param in model.fc.parameters():
  param.requires_grad = False</code>
  </pre>
  <p>You can instead thaw parameters that are frozen by setting <code>requires_grad</code> to <code>True</code>.</p>
  <h3>BatchNorm</h3>
  <p>The <code>BatchNorm</code> layer is a special case: it has two parameters (gamma and beta), but it also has two buffers that are used to accumulate the mean and standard deviation of the dataset during training. If you only use <code>requires_grad=False</code> then you are only fixing gamma and beta. The statistics about the dataset are still accumulated. Sometimes fixing those as well can help the performance, but not always. Experimentation, as usual, is key.</p>
  <p>If we want to also freeze the statistics accumulated we need to put the entire layer in evaluation mode by using <code>eval</code> (instead of <code>requires_grad=False</code> for its parameters):</p>
  <pre><code>model.bn.eval()</code></pre>
  <p>Note that this is different than using <code>model.eval()</code> (which would put the entire model in evaluation mode). You can invert this operation by putting the BatchNorm layer back into training mode: <code>model.bn.train()</code>.</p>
  <h3>Understanding the Architecture We Are Using</h3>
  <p>When doing transfer learning, in many cases you need to know the layout of the architecture you are using so you can decide what to freeze or not to freeze. In particular, you often need either the name or the position of a specific layer in the network.</p>
  <p>
  As usual, we do not encourage the use of
  <code>print(model)</code> as the output there does NOT necessarily correspond to the execution path coded in the
  <code>forward</code> method. Instead, use the documentation of the model or export the model and visualize it with
  <a target="_blank"  href="https://netron.app">Netron</a> as explained in the next subsection.
  </p>
  <h3>Visualizing an Architecture with Netron</h3>
  <p>Netron is a web app, so you do not need to install anything locally. First we need to export the model:</p>
  <pre>
  <code>
    # Fake image needed for torch.jit.trace
    # (adjust the size of the image from 224x224 to what the# network expects if needed)
    random_image = torch.rand((1, 3, 224, 224))>
    scripted = torch.jit.trace(model, random_image)
    torch.jit.save(scripted, "my_network.pt")
  </code>
  </pre>
  <p>Then we can go to <a target="_blank" href='https://netron.app/'>Netron</a> and load this file. Once the architecture has been loaded, press Crtl+U to visualize the name of each layer. For example, this is the last part of a ResNet architecture:</p>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/e7c5e704-990b-4fdf-9f08-0f926ce2c30b'>
  <p>The last layer is called <code>fc</code>. Netron is also telling us that there are 1000 x 512 weights. This means that there are 512 inputs to the layer and 1000 outputs (the ImageNet classes are 1000). If we want to freeze the parameters of that layer we can do:</p>
  <pre>
    <code>
for param in model.fc.parameters():
  param.requires_grad = False</code>
  </pre>
</div>
