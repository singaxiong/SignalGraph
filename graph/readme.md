# Introduction
Each node in the computational graph is called a node or layer. There are input nodes, output nodes, cost function nodes, etc. 
Each node need to have two functions, i.e. the forward pass and backward pass (back-propagation). Some nodes, such as short time fourier transform, only have forward pass function because we don't need the backward pass now. 
This folder contains all the forward (F_xxx) and backward (B_xxx) functions of the nodes, and also some supporting functions. 

#List of nodes
* Input nodes
  * Normal 
* Cost nodes

* Intermediate nodes
  - Elementwise add: output = F_add(prev_layers). Add the activation of all prev_layers, assuming that they all have exactly the same size. 
  - Affine transform: [output, validFrameMask] = F_affine_transform(input_layer, transform, bias). The most common node type of neural networks. 
  - Beamforming: [output] = F_beamforming(input_layers, curr_layer). Apply beamforming filter (frequency domain) to STFT coefficients. 
  - Trajectory mean subtraction: [output,validFrameMask] = F_cmn(input_layer). Subtract the mean of a trajectory from the trajectory itself. E.g. the cepstral mean normalization used in ASR systems. 
  - Compute GCC features: [gcc, maskGCC] = F_comp_gcc(input_layer, curr_layer). Compute generalized cross correlation between micriphones. Only forward pass available. 
  - Concatenate: output = F_concatenate(prev_layers). Concatenate the activations of prev_layers, assuming that the first dimensions are the same. 
  - Convolution: output = F_convolution(input, weights, b). Not finished yet. 
  - Consine distance: output = F_cosine(input_layers). Compute the cosine distance of the activations of the two input layers. Applied on each column vectors independently. 
  - Covariance: output = F_cov(input). Compute the covariance matrix of the input trajectory. 
  - Cross entropy: [cost,acc] = F_cross_entropy(input_layers, CE_layer). The common cross entropy between posterior probabilities and label. Support scaling of each column vector. 
  - Dynamic features: [output,validFrameMask] = F_dynamic_feat(input_layer). Compute the first and second time derivatives of vector sequences. Implemented according to the delta and acceleration features of ASR. 
  - Enframe: output = F_enframe(input, frame_len, frame_shift). Put time domain signal into overlapping frames. Forward pass only. 
  - Extract dimensions: [output] = F_ExtractDims(input_layer, idx). Extract a subset of dimensions from the input, the subset is defined by "idx". 
  - Filter: output = F_filter(input_layers). Apply filter (from the first input layer) to data (from the second input layer). 
