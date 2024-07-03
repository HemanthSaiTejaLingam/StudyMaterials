<div>
  <p>You can play on your own with the CNN Explainer website <a target="_blank" href="https://poloclub.github.io/cnn-explainer/">here<span>(opens in a new tab)</span></a> and verify that what you see there matches your expectations, given what you now know about CNNs.</p>
  <h3>Optional Resources</h3>
  <p>
    If you would like to know more about interpreting CNNs and convolutional layers in particular, you are encouraged to check out these resources:
  </p>
  <h4>What is the network looking at?</h4>
  <p>
    <a href="https://github.com/jacobgil/pytorch-grad-cam" target="_blank">The PyTorch Grad-CAM library</a> implements several methods to interpret the decision of a CNN when classifying an image. It also contains references to the relevant papers.
  </p>
  <h4>Visualizing CNN layers</h4>
  <ul>
    <li>
      Here's the <a href="http://cs231n.github.io/understanding-cnn/" target="_blank">Visualizing CNNs section from the Stanford CS231n course</a> on visualizing what CNNs learn.
    </li>
    <li>
      Here's a <a href="https://www.youtube.com/watch?v=AgkfIQ4IGaM&amp;t=78s" target="_blank">demonstration of a CNN visualization tool on YouTube</a>. If you'd like to learn more about how these visualizations are made, check out this <a href="https://www.youtube.com/watch?v=ghEmQSxT6tw&amp;t=5s" target="_blank">explanatory video on CNN visualizations</a>.
    </li>
    <li>
      Read this <a href="https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html" target="_blank">Keras blog post on visualizing how CNNs see the world</a>, which provides an accessible introduction to Deep Dreams. When you've read that:
      <ul>
        <li>
          Also check out this <a href="https://www.youtube.com/watch?v=XatXy6ZhKZw" target="_blank">music video incorporating Deep Dreams effects on YouTube</a> (look at 3:15-3:40)!
        </li>
        <li>
          Create your own Deep Dreams (without writing any code!) using this <a href="https://deepdreamgenerator.com/" target="_blank">Deep Dream Generator website</a>.
        </li>
      </ul>
    </li>
    <li>
      If you'd like to read more about interpretability of CNNs, here's an <a href="https://openai.com/research/attacking-machine-learning-with-adversarial-examples" target="_blank">OpenAI article on attacking machine learning with adversarial examples</a> that details some dangers from using deep learning models (that are not fully interpretable) in real-world applications.
    </li>
  </ul>
</div>
<div>
    <h2>Visualizing CNNs</h2>
    <p>Let’s look at a sample CNN that has been pre-trained on ImageNet to see how it works in action. This is an example of an architecture for a transfer learning use case.</p>
    <p>Let's verify that what we have discussed in this lesson matches what we see, in particular the fact that the first layers focus on edges and low-level features and can therefore be recycled even on different datasets.</p>
    <p>The CNN we will look at is trained on ImageNet as described in <a target="_blank" href="https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf">Zeiler and Fergus's paper on visualizing and understanding convolutional networks<span>(opens in a new tab)</span></a>. In the images below (from the same paper), we’ll see <em>what</em> each layer in this network detects and see <em>how</em> each layer detects more and more complex ideas.</p>
</div>
