Shady Neural Network
====================

Shady Neural Network is a rust framwork for running, creating, and training
neural networks. The name "Shady" Neural Network comes from the use of shaders
in all the matrix operation functionality.

## The Math
Starting off, lets take a look at the math behind forward propogation. Sending
data through the network is relatively straight forward. For every layer we
take our inputs and multiply by a weight matrix which describes which weight
is assined to which input for each node in the layer. We can represent the weight matrix using:
[<img src="/images/weights.png" height="20"/>](/images/weights.png)
