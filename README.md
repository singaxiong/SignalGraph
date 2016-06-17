# SignalGraph

Author: Xiong Xiao, Nanyang Technological University, Singapore

Date created: 16 Jun 2016

Email: xiao.xiong.1981@gmail.com / xiaoxiong@ntu.edu.sg

SignalGraph is a Matlab-based tool for building arbitrary directed acyclic graphs (DAG) for signal processing. The original purpose is to make it easy to apply deep learning techniques on speech signals on the Matlab platform. It should also be applied to other tasks, especially involving temporal trajectory data. 

The tool now supports several common types of neural networks, such as feedforward deep neural networks (DNN), long-short term memory (LSTM) recurrent neural network (RNN), and convolutional neural network (CNN, currently only support temporal convolution). It allows arbitrary connections between the layers as far as there is no loop (recurrent connections). Recurrency is allowed within a layer, such as in a LSTM layer, but not allowed between layers. 

The tool also supports common signal processing modules, such as extracting of MFCC features, overlap-and-add for speech enhancement, extraction of generalized cross correlation (GCC) from multi-channel waveforms, etc. Matlab is a great platform for signal processing. One of the major goal of this tool is to promote deep learning in classic signal processing areas. More signal processing functions and neural network types will be added in the future. 

Examples of using the tool for various purposes are provided, e.g. acoustic modeling for speech recognition, denoising speech, implementing beamforming using neural networks, fixed-dimensional embedding of sequential data, etc. Some of these examples will be added later as I am still working on them. I encourage those using this tool to publish your recipe in the examples/ directory so we can form a community and share the latest research results. 

Things to do:

1. Documentation
1.1 Add slides to introduce the tool, including the architecture, how to define graph structure, how to define input streams, how to define cost functions, etc. 
1.2 Add comments to functions

2. Better design of the achitecture
2.1 Streamline the processing of input streams

3. Examples to building various network types, such as DNN, CNN, and LSTM. 
3.1 Joint estimation of beamforming network and acoustic model network
3.2 DOA estimation using neural networks
3.3 Speech dereverberation/denoising using DNN/LSTM
