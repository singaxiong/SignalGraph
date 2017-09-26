# SignalGraph


Updates

Add data simulation example that uses the imaging method to generate room impulse response (RIR) and simulate reverberant and noisy speech signals for microphone array. 

Author: Xiong Xiao, Nanyang Technological University, Singapore

Date created: 16 Jun 2016

Email: xiao.xiong.1981@gmail.com / xiaoxiong@ntu.edu.sg

Lastest update

Added examples of LSTM based classification task (acoustic modeling on TIMIT) (03 Aug 2016)

Added examples for DNN based regression and classification tasks (16 Jun 2016)


Introduction

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

Papers based on SignaGraph:

[1] Xiong Xiao, Shengkui Zhao, Duc Hoang Ha Nguyen, Xionghu Zhong, Douglas L. Jones, Eng Siong Chng, Haizhou Li, "Speech dereverberation for enhancement and recognition using dynamic features constrained deep neural networks and feature adaptation", EURASIP Journal on Advances in Signal Processing, 2016(1), pp. 1-18. 

[2] Xiong Xiao, Shinji Watanabe, Hakan Erdogan, Liang Lu, John Hershey, Michael L. Seltzer, Guoguo Chen, Yu Zhang, Michael Mandel, Dong Yu, "Deep beamforming networks for multi-channel speech recognition", in ICASSP 2016.

[3] Zeyan Oo, Yuta Kawakami, Longbiao Wang, Seiichi Nakagawa, Xiong Xiao and Masahiro, "DNN-based Amplitude and Phase Feature Enhancement for Noise Robust Speaker Identification", accepted by INTERSPEECH 2016. 

[4] Xiaohai Tian, Zhizheng Wu, Xiong Xiao, Eng Siong Chng, Haizhou Li, "Spoofing detection from a feature representation perspective", in ICASSP 2016.

[5] Xiaohai Tian, Zhizheng Wu, Xiong Xiao, Eng Siong Chng, Haizhou Li, "An investigation of spoofing speech detection under additive noise and reverberant conditions", accepted by INTERSPEECH 2016. 

[6] Jia Yu, Xiong Xiao, Lei Xie, Eng Siong Chng and Haizhou Li, "A DNN-HMM Approach to Story Segmentation", accepted by INTERSPEECH 2016. 

[7] Xiong Xiao, Shengkui Zhao, Xionghu Zhong, Douglas L. Jones, Eng Siong Chng, Haizhou Li, "Learning to Estimate Reverberation Time in Noisy and Reverberant Rooms", in proceedings of InterSpeech 2015.

[8] Xiong Xiao, Shengkui Zhao, Xionghu Zhong, Douglas L. Jones, Eng Siong Chng, Haizhou Li, "A learning-based approach to direction of arrival estimation in noisy and reverberant environments", in ICASSP 2015.
Steven Du, Xiong Xiao, Eng Siong Chng, "DNN feature compensation for noise robust speaker verification", in proceedings of ChinaSIP 2015. 

[9] Xiong Xiao, Xiaohai Tian, Steven Du, Haihua Xu, Eng Siong Chng, Haizhou Li, "Spoofing Speech Detection Using High Dimensional Magnitude and Phase Features: the NTU Approach for ASVspoof 2015 Challenge", in proceedings of InterSpeech 2015. 

[10] Guangpu Huang, Chenglin Xu, Xiong Xiao, Lei Xie, Eng Siong Chng, Haizhou Li "Multi-View Features in a DNN-CRF Model for Improved Sentence Unit Detection on English Broadcast News”, accepted by APSIPA 2014.

[11] Chenglin Xu, Lei Xie, Guangpu Huang, Xiong Xiao, Eng Siong Chng, Haizhou Li, "A Deep Neural Network Approach for Sentence Boundary Detection in Broadcast News", in proceedings of Interspeech 2014.

[12] Xiong Xiao, Shengkui Zhao, Duc Hoang Ha Nguyen, Xionghu Zhong, Douglas L. Jones, Eng Siong Chng, Haizhou Li, "The NTU-ADSC systems for Reverberation Challenge 2014", in Reverberation Challenge Workshop 2014.
