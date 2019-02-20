# GAN Network to add or remove attribute from faces
We have have created a GAN in which facial attributes e.g. sunglasses, hats get inverted i.e.
a person without a hat gets a hat. The GAN uses supervised learning and requires an input
of data with and data without the attribute. The GAN is non-standard in its properties.
Instead of having one generator we have two. One that creates a positive attribute out of
a negative, and one that creates a negative picture out of a positive.

The trained network can then be tested with a webbapp using TensorFlow.js.

We used the architechture proposed by https://arxiv.org/abs/1612.05363
