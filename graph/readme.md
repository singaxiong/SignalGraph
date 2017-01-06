# Introduction
Each node in the computational graph is called a node or layer. There are input nodes, output nodes, cost function nodes, etc. 
Each node need to have two functions, i.e. the forward pass and backward pass (back-propagation). Some nodes, such as short time fourier transform, only have forward pass function because we don't need the backward pass now. 
This folder contains all the forward (F_xxx) and backward (B_xxx) functions of the nodes, and also some supporting functions. 

#List of nodes
* Input nodes
  - Input: no function associated. Allow the network to input  
  - Weight to activation: curr_layer = F_weight2activation(curr_layer). This is a specially designed node type. It's activation is the same as its weight. This node is used when we want to have the weights as inputs. 
  
* Cost nodes
  - Consine distance: output = F_cosine(input_layers). Compute the cosine distance of the activations of the two input layers. Applied on each column vectors independently. 
  - Mean square error: cost = F_mean_square_error(input_layers, useMahaDist, CostLayer). MSE cost function for regression. 
  - Logistic: [cost, acc] = F_logistic(input_layers, CostLayer). The logistic cost function for two class classification problems. 
  - Cross entropy: [cost,acc] = F_cross_entropy(input_layers, CE_layer). The common cross entropy between posterior probabilities and label. Support scaling of each column vector. 
  - Multi cross entropy: [cost,acc] = F_multi_cross_entropy(input_layers, CE_layer). Multiple cross entropy cost function. Useful for simultaneously predicting multiple classification targets. 
  - GMM log likelihood: curr_layer = F_ll_gmm(input, curr_layer). Compute the log likelihood of the input vectors on a given GMM model. 
  - Log determinant: output = F_logdet(input). Compute the log determinant of input matrix, usually a covariance matrix. 
  - Joint cost: cost = F_jointCost(input_layer, curr_layer). Compute the joint cost between neighboring frames. This can be used to measure the smoothness of vector trajectories. 

* Common neural network nodes
  - Affine transform: [output, validFrameMask] = F_affine_transform(input_layer, transform, bias). The most common node type of neural networks. 
  - Sigmoid: [output] = F_sigmoid(input_layer). Apply sigmoid activation elementwise. 
  - Tanh: [output] = F_tanh(input_layer). Apply tanh activation. 
  - Temporal convolution: [output,X2] = F_tconv(input, curr_layer). Apply temporal convolution. 
  - Temporal max pooling: [output,idx,validFrameMask] = F_tmaxpool(input_layer, curr_layer). Apply temporal max pooling. Used together with temporal convolution. 
  - Word to vector: output = F_word2vec(input, W, singlePrecision). Apply transform W on sparse input (one hot). 
  - Convolution: output = F_convolution(input, weights, b). Not finished yet. 
  - LSTM: [LSTM_layer] = F_LSTM(input_layer, LSTM_layer). A single unidirectional LSTM layer. 
  - Softmax: [output] = F_softmax(input_layer). Apply softmax activation. 
  - Multi softmax: output = F_multi_softmax(input_layer, TaskVocabSizes). Useful for multiple classification tasks. 

* Signal processing nodes
  - Enframe: output = F_enframe(input, frame_len, frame_shift). Put time domain signal into overlapping frames. Forward pass only. 
  - Short time Fourier transform: [fft_x, maskFFT] = F_stft(input_layer, curr_layer). Extract STFT from input signals. Forward only. Allow control over window type, frame length, frame shift, fft length, whether to use DC removal, etc. 
  - Compute GCC features: [gcc, maskGCC] = F_comp_gcc(input_layer, curr_layer). Compute generalized cross correlation between micriphones. Only forward pass available. 
  - Beamforming: [output] = F_beamforming(input_layers, curr_layer). Apply beamforming filter (frequency domain) to STFT coefficients. 
  - Filter: output = F_filter(input_layers). Apply filter (from the first input layer) to data (from the second input layer). 
  - Itakura Saito distance: output = F_Itakura_Saito(input_layers). Compute the IS distance between clean speech and noisy/enhanced speech vectors. 
  - Logarithm: [output,validFrameMask] = F_log(input_layer, const). Apply logarithm to each element. Option to add a small constant for stable gradient. 
  - MVDR beamforming with spatial covariance matrices: curr_layer = F_MVDR_spatialCov(input_layer, curr_layer). Estimate MVDR beamforming parameters from noise and speech spatial covariance matrices (SCM). 
  - Spatial covariance: output = F_SpatialCov(input_layer, curr_layer). Compute the spatial covariance of input. 
  - Spatial covariance with mask: output = F_SpatialCovMask(prev_layers, curr_layer). Compute the spatial covariance matrices (SCM) of input with the help of a mask. Produce both noise and speech SCMs. 
  - Spatial covariance with split mask: output = F_SpatialCovSplitMask(prev_layers, curr_layer). Same as SpatialCovMask, but now uses separate speech and noise masks. 
  - TDOA to beamforming weight: output = F_tdoa2weight(input, freq_bin). Convert TDOA to beamforming weight. need to supply freq_bin which tells us which frequency bins to use. 
  - Power spectrum: [output] = F_power_spectrum(input_layer). Compute power spectrum from complex Fourier coefficients. 
  - Power spectrum split: output = F_power_spectrum_split(input). Compute power spectrum from concatenated real and imaginary elements of Fourier coefficients. 
  - Real and imaginary parts to beamforming weights: [output,validFrameMask] = F_real_imag2BFweight(input_layer, freq_bin, online). Convert the real and imaginary parts (concatenated) to the corresponding complex numbers of beamforming filter. Obsolute, see realImag2complex. 
  - Real and imaginary parts to complex number: [output] = F_realImag2complex(input_layer). Convert the real and imaginary parts (concatenated) to the corresponding complex numbers. 
  - Covariance: output = F_cov(input). Compute the covariance matrix of the input trajectory. 
  - Dynamic features: [output,validFrameMask] = F_dynamic_feat(input_layer). Compute the first and second time derivatives of vector sequences. Implemented according to the delta and acceleration features of ASR. 

* Data manipulation nodes
  - Concatenate: output = F_concatenate(prev_layers). Concatenate the activations of prev_layers, assuming that the first dimensions are the same. 
  - Transpose: [output] = F_transpose(input_layer, curr_layer). Transpose input matrix. 
  - Repeat a matrix: [output] = F_repmat(input_layer, curr_layer). Similar to repmat of Matlab. 
  - Reshape a matrix: [output] = F_reshape(input_layer, curr_layer). Similiar to reshape of Matlab. 
  - Permute: output = F_permute(input_layer, curr_layer). Change the order of the data dimensions. 
  - Splice: [output,validFrameMask] = F_splice(input_layer, context). Concatenate neighbouring frames to incorporate context information. For example, splice feature vectors before DNN based acoustic model. 
  - Extract dimensions: [output] = F_ExtractDims(input_layer, idx). Extract a subset of dimensions from the input, the subset is defined by "idx". 
  - Frame select: [output, validFrameMask] = F_frame_select(input_layer, curr_layer). Select certain column vectors (frames) of the input. 
  - Frame shift: [output, validFrameMask] = F_frame_shift(input_layer, curr_layer). Shift the column vectors (frames) of the input. 

* Mathematical nodes
  - Elementwise add: output = F_add(prev_layers). Add the activation of all prev_layers, assuming that they all have exactly the same size. 
  - Max: output = F_max(input_layer, curr_layer). Elementwise max function. 
  - Mean: output = F_mean(input_layer, curr_layer). Elementwise mean pooling. 
  - Median: output = F_median(input_layer, curr_layer). Elementwise median pooling. 
  - Min: output = F_min(input_layer, curr_layer). Elementwise min pooling. 
  - Elementwise multiplication: [output, validFrameMask] = F_hadamard(input_layers). Multiply the activation of previous layers element-by-element. 
  - Trajectory mean subtraction: [output,validFrameMask] = F_cmn(input_layer). Subtract the mean of a trajectory from the trajectory itself. E.g. the cepstral mean normalization used in ASR systems. 
  - Index to vector: [output,validMask] = F_idx2vec(input_layer, curr_layer, single_precision). Convert class label into one-hot representation. 
  - Inner product: output = F_inner_product(input_layers). Compute inner product of two vectors. Applied to column vectors independently. 
  - Inner product normalized: output = F_inner_product_normalized(input_layers). Same as inner product, but normalized by the size of column vectors. 
  - Linear discriminant analysis: not finished. 
  - Sparse linear transform: output = F_sparse_affine_transform(input, transform, bias, singlePrecision). Specially implemented for sparse input for speedup. Usually used for word embedding in NLP tasks. 
  - Weighted average: [output, weights] = F_weighted_average(prev_layers). Obtain weighted average of the input. One previous layer provide the weight and the other provide the input data. 
  - Weighting: output = F_weighting(input, weight, bias). Not finished. 
  - Within Covariance: output = F_Within_Cov(input_layers). Not finished. 
