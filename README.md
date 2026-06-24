1.) Week 1: Foundations of Neural Networks
 The Big Picture: Getting comfortable with how a basic neural network learns non-linear relationships by pairing simple linear equations with non-linear activation  functions.
What I Focused On:
-Activation Functions: Prcticed using ReLU in hidden layers to handle non-linearity (by turning negative values to zero with f(x) = max(0, x)) and Softmax at the very end to turn raw model outputs into clean probabilities.
Network Arcitecture: Wrapped my head around how data moves from input layers through hidden layers, and how weights and biases scale and shift that information along the way.

2.)Week 2: Backpropagation & Optimization
The Big Picture: 
Learning the actual mechanics behind how a network catches its own mistakes and updates its parameters to get smarter.
What I Focused On:
-The Chain Rule: Spent time calculating partial derivtives of the Loss Function with respect to individual weights (dLoss/dw) and biases (d Loss /db)
-Gradient Descent: Visualized how optimization steps down the cost function curve toward a global minimum, and how to keep the model from getting stuck or blowing up.

3.) Week 3: Advanced Training & Loss Functions
The Big Picture: Learning how to fine-tune training settings and pick the right evaluation strategy for classification tasks.
What I Focused On:
Cross-Entropy Loss: Implemented Cross-Entropy to see how closely our model's predicted probabilities actually match up against the true labels.
Step Size Tuning: Experimented with how learning rates make or break a model—finding that sweet spot where training converges smoothly without overshooting or taking forever.

4.)Week 4: Convolutional Neural Networks (CNNs) – Part 1
The Big Picture: Moving away from standard dense networks and building networks that actually understand spatial relationships in images.
What I Focused On:
Convolution Layers: Using filers/kernels to scan an image and pick up on low-level features like sharp edges, lines, and gradients.
Pooling Layers: Implementing Max Pooling to shrink the size of our feature maps. This cuts down on computing power but keeps the most important visual features intact.

5.)Week 5: Convolutional Neural Networks (CNNs) – Part 2
The Big Picture: Puting all the pieces together into a complete image classification pipeline.
What I Focused On:
Flattening & Dense Layers: Unrolling 2D pooled maps into a single 1D vector so the spatial features can finally be fed into a regular dense layer.
Classification Output: Mapping those final features to a Softmax layer to spit out the final predicted class probabilities.
