# Project Introduction

This repository is dedicated to the project developed by four Applied Mathematical Engineering students as part of their final thesis at Queen's University. The Custom Compression class developed as well as the final network architecture are in the following files:

- [Custom Compression Class](CustomCompression.py)
- [Final Network Architecture](NNA.py)

Be sure to explore the full research paper for more details:

[Check out the full research paper (PDF)](Data%20Compression%20via%20Nonlinear%20Transform%20Coding%20using%20Artificial%20Neural%20Networks.pdf)


## Abstract

This thesis presents an implementation of nonlinear transform coding (NTC) that utilizes artificial neural networks (ANNs) for lossy data compression. In today's digital age, efficient data compression is of paramount importance. The reliance on multimedia information spans across various industries, including healthcare. Our project specifically focuses on the compression of computerized tomography (CT) scans and examines the societal, economic, and environmental impacts that improvements in compression could have.

Common image compression techniques typically rely on linear transform coding, using orthogonal transformations to decorrelate and compress a source. NTC, on the other hand, offers a more sophisticated approach but has been limited historically by the complexity of determining suitable nonlinear transforms in high-dimensional spaces. Recent advancements in computer hardware and the emergence of ANNs have provided a means to implement general nonlinear transforms, driving significant research in the field of NTC.

We provide a mathematical formulation of an NTC system, consisting of nonlinear analysis and synthesis transforms implemented using ANNs. We derive a differentiable cost function, which involves approximations of the source entropy model and a proxy for scalar quantization. This cost function can be optimized using stochastic gradient descent and, with the incorporation of Lagrangian optimization, can be used for rate-distortion traversal. An iterative design process is employed to enhance the model's compression capabilities, incorporating techniques such as image tiling, regularization techniques, and more sophisticated distortion measures and activation functions, namely the structural similarity index measure and generative divisive normalization, respectively.
