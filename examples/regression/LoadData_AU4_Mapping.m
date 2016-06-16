function [Data_tr, Data_cv, para] = LoadData_AU4(para)
load('../data/Aurora4/mfcc_label.mat');

Data_tr(1).data = mfcc_noisy_train;
Data_cv(1).data = mfcc_noisy_test;

Data_tr(2).data = mfcc_clean_train;
Data_cv(2).data = mfcc_clean_test;

para.IO.context = [9 1];                    	% the context of input and output streams. Here we use 9 contextual frames for MFCC

end
