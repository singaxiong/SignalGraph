This directory shows how to perform speech dereverberation and speech enhancement. To run the examples, you will need the REVERB Challenge corpus, which can be downloaded from http://reverb2014.dereverberation.com/download.html

1. TrainDereverbNet_Regression.m shows how to use LSTM or DNN to implement regression based speech enhancement. That is, the distorted log spectrogram is mapped to clean log spectrogram. 

Things to do: 
Time frequency masking based speech enhancement. 
Temporal Filter prediction based speech dereverberation. 
