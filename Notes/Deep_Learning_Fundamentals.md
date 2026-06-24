My Deep Learning and CNN Study Notes (StatQuest Style)

Neural Networks and Non-Linearity
The whole point of a neural network is to fit weird, curved data. To do that, it takes basic linear equations and passes them through non-linear activation functions. If you don't use activation functions, the whole network just simplifies down to a single boring line, no matter how many layers you stack.

My Main Takeaways on Activations
ReLU: This is super simple but incredibly powerful. It leaves positive numbers completely alone, but flatlines any negative number to 0. That sudden bend at zero is what lets the network map complex curves.
Softmax: I use this at the absolute end of a classification network. It takes whatever raw scores the model spits out and turns them into clean percentages that add up to 100 percent. This gives me actual probabilities for each class.

Wrapping My Head Around Backpropagation
Backpropagation is just the network looking at its mistakes and figuring out how to fix them. It uses the Chain Rule from calculus to walk backward through the layers. The goal is to see how much a tiny change in a specific weight or bias would change the final error.

The Core Math Goals
We calculate how much changing a weight messes with the total loss, and how much changing a bias messes with the total loss.
Once Gradient Descent has these numbers, it tweaks the weights and biases by taking a small step, scaled by our Learning Rate, down the curve until the loss is as close to zero as possible.

Cross-Entropy Loss
When dealing with classification, we can't just use standard distance errors. We use Cross-Entropy Loss. It essentially checks how confident the model is compared to the actual right answer. If the model is 99 percent confident about the wrong answer, Cross-Entropy penalizes it aggressively.

Convolutional Neural Networks (CNNs)
Regular networks smash images into one long flat line of numbers, which completely ruins the spatial layout because the network forgets which pixels are next to each other. CNNs fix this by keeping the image as a 2D grid.

The Core Steps

Convolution: You slide a tiny grid called a filter or kernel across the image. It does quick calculations to flag where things like sharp edges, vertical lines, or corners are.

Max Pooling: This is for downsampling. You slide a window across your new feature map and keep only the biggest number in that window. It cuts the data size in half but keeps the strongest visual features.

Flattening: Once the image is tiny and full of features, you unroll it back into a 1D line so a standard dense layer can finally categorize it.
