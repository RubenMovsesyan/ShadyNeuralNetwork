Shady Neural Network
====================

Shady Neural Network is a rust framwork for running, creating, and training neural networks. The name "Shady" Neural Network comes from the use of shaders in all the matrix operation functionality.

## The Math
### Forward Propogation
Starting off, lets take a look at the math behind forward propogation. Sending data through the network is relatively straight forward. For every layer we take our inputs and dot product by a weight matrix which describes which weight is assined to which input for each node in the layer. We can represent the weight matrix using:

[<img src="/images/weights.png" height="50"/>](/images/weights.png)

and then we add a bias vector to the result of the dot product which we can represent using:

[<img src="/images/biases.png" height="50"/>](/images/biases.png)

from there we run processed values through an activation function that is usually used to introduce non-linearity. We can represent this activation function using the following:

[<img src="/images/activation.png" height="50"/>](/images/activation.png)

where `z` is the processes values of the layer before running them through the activation function.

Putting all this together we can find that our final computation for any given layers feed forward propogation is:

[<img src="/images/feed_forward.png" height="50"/>](/images/feed_forward.png)

where `o` is the output of this layer and `i` is the input to this layer.
