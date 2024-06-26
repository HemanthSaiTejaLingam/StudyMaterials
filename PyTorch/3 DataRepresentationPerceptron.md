<h4>Data Representation</h4>
<div>Rarely can we use "out of the box" input. We need our input to be tensors, but often our raw data consists of images, text, or tabular data, and we can't easily input those directly into our model.
<ul>
<li>For image data, we need the data to be turned into tensors with entries of the tensors as bit values in color channels (usually red, green, and blue).</li>
<li>Text data needs to be tokenized, meaning, individual words or groups of letters need to be mapped to a token value.</li>
<li>For tabular data, we have categorical values (high, medium, low, colors, demographic information, etc...) that we need to transform into numbers for processing.</li><ul></ul></div>
<h4>One-Hot Encoding</h4>
<p>Categorical values can become numbers. For instance, if you have three colors, Red, Blue, and Yellow, you can assign binary values representing whether the color is present or not. The model can then easily compute how far two points are from each other rather than simply assigning arbitrary values (like 1, 2, and 3 to the colors).</p>
<p>
<i>Note: One-Hot Encoding adds columns, which increases the dimensions of our tensors. If you have a lot of categorical features, this makes it even more complicated.</i></p>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/c9d9877b-2197-4455-acaf-bd7235807671'>
<h4>Transforming Data for Neural Networks</h4>
<p>Often, we are faced with data that is not in a format conducive to use in neural networks in its raw form. Preprocessing is the act of turning data from that raw form into tensors that can be used as input to a neural network. This includes:</p>
<ul>
  <li>Encoding non-numerical features</li>
  <li>Converting images to tensors of bit values in color channels</li>
  <li>Tokenizing words</li>
</ul>
<h4>Defining Error Functions</h4>
<div>
An error function is simply a function that measures how far the current state is from the solution. We can calculate the error and then make a change in an attempt to reduce the error—and then repeat this process until we have reduced the error to an acceptable level.
</div>
<h4>Discrete and Continuous Errors</h4>
<div>
  <p>
One approach to reducing errors might be to simply count the number of errors and then make changes until the number of errors is reduced. But taking a discrete approach like this can be problematic—for example, we could change our line in a way that gets closer to the solution, but this change might not (by itself) improve the number of misclassified points.</p>
<p>Instead, we need to construct an error function that is continuous. That way, we can always tell if a small change in the line gets us closer to the solution. We'll do that in this lesson using the log-loss error function. Generally speaking, the log-loss function will assign a large penalty to incorrectly classified points and small penalties to correctly classified points. For a point that is misclassified, the penalty is roughly the distance from the boundary to the point. For a point that is correctly classified, the penalty is almost zero.</p>
<p>We can then calculate a total error by adding all the errors from the corresponding points. Then we can use gradient descent to solve the problem, making very tiny changes to the parameters of the line in order to decrease the total error until we have reached an acceptable minimum.
</p>
<p>
We need to cover some other concepts before we get into the specifics of how to calculate our log-loss function, but we'll come back to it when we dive into gradient descent later in the lesson.
  </p>
</div>
<h4>Perceptrons</h4>
<p>A perceptron is a major building block of neural networks. Perceptrons are graphs that have nodes and edges. In the general case, they look like this:</p>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/aa15f97f-f253-4d92-83ee-a1fb2714d2a8'>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/a8c2b62d-91b8-4f83-b6cf-66746e8ba07f'>
<div>
  <p>A neural network, with three input units, two hidden nodes, and one output layer. Lines connect each input with each hidden node. Each line is labeled with w_ij indices.</p>
<p>Before, we were able to write the weights as an array, indexed as w_i.</p>
<p>But now, the weights need to be stored in a matrix, indexed as w_ij</p>
​
<p>Each row in the matrix will correspond to the weights leading out of a single input unit, and each column will correspond to the weights leading in to a single hidden unit. For our three input units and two hidden units, the weights matrix looks like this:</p>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/a050e651-42da-45cb-8f31-fe18cf9d71e8'>
</div>
<div>
  <p>To initialize these weights in NumPy, we have to provide the shape of the matrix. If features is a 2D array containing the input data:</p>
<b>
## Number of records and input units<br>
n_records, n_inputs = features.shape<br>
## Number of hidden units<br>
n_hidden = 2<br>
weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))<br></b>
<p>This creates a 2D array (i.e. a matrix) named weights_input_to_hidden with dimensions n_inputs by n_hidden. Remember how the input to a hidden unit is the sum of all the inputs multiplied by the hidden unit's weights. So for each hidden layer unit h_j,</p> 

<p>we need to calculate the following:</p>
<b> ℎ_j=\sum_{i=1} w_ij*x_i </b><br>
<p>In this case, we're multiplying the inputs (a row vector here) by the weights. To do this, you take the dot (inner) product of the inputs with each column in the weights matrix. For example, to calculate the input to the first hidden unit,j=1, you'd take the dot product of the inputs with the first column of the weights matrix, like so:</p>
<b>h_1=x_1*w_11+x_2*w_21+x_3*w_31</b><br>
  <img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/f86d5f70-82b4-4198-b65c-c4b03e468c69'>
</div>
<div>
Calculating the input to the first hidden unit with the first column of the weights matrix.<br>
And for the second hidden layer input, you calculate the dot product of the inputs with the second column. And so on and so forth.<br>
In NumPy, you can do this for all the inputs and all the outputs at once using np.dot<br>
<b>hidden_inputs = np.dot(inputs, weights_input_to_hidden)</b>
</div>
<img src='https://github.com/HemanthSaiTejaLingam/StudyMaterials/assets/114983155/a813c0a2-5c39-4fc4-8b5a-469fb5cd73d7'>
<div>
  <h4>Making a column vector</h4>
<p>You see above that sometimes you'll want a column vector, even though by default NumPy arrays work like row vectors. It's possible to get the transpose of an array like so arr.T, but for a 1D array, the transpose will return a row vector. Instead, use arr[:,None] to create a column vector:
</p>
print(features)<br>
array([ 0.49671415, -0.1382643 ,  0.64768854])

print(features.T)<br>
array([ 0.49671415, -0.1382643 ,  0.64768854])

print(features[:, None])<br>
array([[ 0.49671415],
       [-0.1382643 ],
       [ 0.64768854]])

<p>Alternatively, you can create arrays with two dimensions. Then, you can use arr.T to get the column vector.<p></p><br>


np.array(features, ndmin=2)<br>
array([[ 0.49671415, -0.1382643 ,  0.64768854]])

np.array(features, ndmin=2).T<br>
array([[ 0.49671415],
       [-0.1382643 ],
       [ 0.64768854]])
I personally prefer keeping all vectors as 1D arrays, it just works better in my head.
</div>
<div>
  <h4>Backpropagation</h4>
<p>
Now we've come to the problem of how to make a multilayer neural network learn. Before, we saw how to update weights with gradient descent. The backpropagation algorithm is just an extension of that, using the chain rule to find the error with the respect to the weights connecting the input layer to the hidden layer (for a two layer network).
</p>
<p>
To update the weights to hidden layers using gradient descent, you need to know how much error each of the hidden units contributed to the final output. Since the output of a layer is determined by the weights between layers, the error resulting from units is scaled by the weights going forward through the network. Since we know the error at the output, we can use the weights to work backwards to hidden layers.
</p>
<p>
For example, in the output layer, you have errors 𝛿_k^0 attributed to each output unit k. Then, the error attributed to hidden unit j is the output errors, scaled by the weights between the output and hidden layers (and the gradient)</p>
Then, the gradient descent step is the same as before, just with the new errors
​</div>
