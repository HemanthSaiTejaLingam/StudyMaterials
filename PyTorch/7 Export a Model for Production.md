<div>
  <h2>Export a Model for Production</h2>
  <h3>Wrap Your Model for Inference</h3>
  <p>Exporting a model for production means packaging your model in a stand-alone format that can be transferred and used to perform inference in a production environment, such as an API or a website.</p>
  <h3>Production-Ready Preprocessing</h3>
  <p>Remember that the images need some preprocessing before being fed to the CNN. For example, typically you need to resize, center crop, and normalize the image with a transform pipeline similar to this:</p>
  <pre>
<code>testval_transforms = T.Compose(
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
  <p>Obviously, if you do not do these operations in production the performance of your model is going to suffer greatly.</p>
  <p>The best course of action is to make these transformations part of your standalone package instead of re-implementing them in the production environment. Let's see how.</p>
  <p>We need to wrap our model in a wrapper class that is going to take care of applying the transformations and then run the transformed image through the CNN.</p>
  <p>If we trained with the <code>nn.CrossEntropyLoss</code> as the loss function, we also need to apply a softmax function to the output of the model so that the output of the wrapper will be probabilities and not merely scores.</p>
  <p>Let's see an example of such a wrapper class:</p>
  <pre>
    <code>
import torch
from torchvision import datasets
import torchvision.transforms as T
from __future__ import annotations
class Predictor(nn.Module):
    def __init__(
      self, 
      model: nn.Module, 
      class_names: list[str], 
      mean: torch.Tensor, 
      std: torch.Tensor
    ):
        super().__init__()
        self.model = model.eval()
        self.class_names = class_names
        self.transforms = nn.Sequential(
            T.Resize([256, ]),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # 1. apply transforms
            x = self.transforms(x)  # =
            # 2. get the logits
            x = self.model(x)  # =
            # 3. apply softmax
            #    HINT: remmeber to apply softmax across dim=1
            x = F.softmax(x, dim=1)  # =
            return x</code>
  </pre>
  <h3>The Constructor</h3>
  <p>Let's first look at the constructor <code>__init__</code>: we first set the model to eval mode, and we also save the class names. This will be useful in production: the wrapper will return the probability for each class, so if we take the maximum of that probability and take the corresponding element from the <code>class_names</code> list, we can return the winning label.</p>
  <p>Then we have this:</p>
  <pre>
<code>
  self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )</code>
  </pre>
  <p>This defines the transformations we want to apply. It looks very similar to the transform validation pipeline, with a few important differences:</p>
  <ul>
    <li>We do not use <code>nn.Compose</code> but <code>nn.Sequential</code>. Indeed the former is not supported by <code>torch.script</code> (the export functionality of PyTorch).</li>
    <li>In <code>Resize</code> the size specification must be a tuple or a list, and not a scalar as we were able to do during training.</li>
    <li>There is no ToTensor. Instead, we use <code>T.ConvertImageDtype</code>. Indeed, in this context the input to the forward method is going to be already a Tensor</li>
  </ul>
  <h3>The <code>forward</code> Method</h3>
  <p>Let's now look at the <code>forward</code> method:</p>
  <pre>
    <code>
def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # 1. apply transforms
            x = self.transforms(x)  # =
            # 2. get the logits
            x = self.model(x)  # =
            # 3. apply softmax
            #    HINT: remmeber to apply softmax across dim=1
            x = F.softmax(x, dim=1)  # =
            return x</code>
  </pre>
  <p>We declare we are not going to need gradients with the <code>torch.no_grad</code> context manager. Then, as promised, we first apply the transforms, then we pass the result through the model, and finally we apply the softmax function to transform the scores into probabilities.</p>
  <h3>Export Using <code>torchscript</code></h3>
  <p>We can now create an instance of our <code>Predictor</code> wrapper and save it to file using <code>torch.script</code>:</p>
  <pre>
    <code>
predictor = Predictor(model, class_names, mean, std).cpu()
# Export using torch.jit.script
scripted_predictor = torch.jit.script(predictor)
scripted_predictor.save("standalone_model.pt")</code>
  </pre>
  <p>Note that we move the Predictor instance to the CPU before exporting it. When reloading the model, the model will be loaded on the device it was taken from. So if we want to do inference on the CPU, we need to first move the model there. In many cases CPUs are enough for inference, and they are much cheaper than GPUs.</p>
  <p>We then use <code>torch.jit.script</code> which converts our wrapped model into an intermediate format that can be saved to disk (which we do immediately after).</p>
  <p>Now, in a different process or a different computer altogether, we can do:</p>
  <pre>
    <code>
import torch
predictor_reloaded = torch.jit.load("standalone_model.pt")</code>
  </pre>
  <p>This will recreate our wrapped model. We can then use it as follows:</p>
  <pre>
    <code>
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
# Reload the model
learn_inf = torch.jit.load("standalone_model.pt")
# Read an image and transform it to tensor to simulate what would
# happen in production
img = Image.open("static_images/test/09.Golden_Gate_Bridge/190f3bae17c32c37.jpg")
# We use .unsqueeze because the model expects a batch, so this
# creates a batch of 1 element
pil_to_tensor = T.ToTensor()(img).unsqueeze_(0)
# Perform inference and get the softmax vector
softmax = predictor_reloaded(pil_to_tensor).squeeze()
# Get index of the winning label
max_idx = softmax.argmax()
# Print winning label using the class_names attribute of the 
# model wrapper
print(f"Prediction: {learn_inf.class_names[max_idx]}")</code>
  </pre>
  <p>NOTE that there are 2 different formats that can be used to export a model: <a target="_blank" rel="noopener noreferrer" href="https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script">script</a> and <a target="_blank" rel="noopener noreferrer" href="https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace">trace</a>. Scripting is more general, but in some cases you do have to use tracing.</p>
</div>
