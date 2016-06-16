% Generate a block of data for training neural networks or inference.
% Organization of data:
%   All data: the whole set of training/dev data
%   Block: a subset of all data that can be easily fit into system memory.
%   minibatch: a fixed size number of training examples for updating the
%       model parameters.
%
% This function support:
%   1. Extracting a subset of all data using predefined indexes.
%   2. Dynamically generating a block of training data by
%       2.1 adding noise and reverberation to clean speech with randomly
%       selected noise sample, SNR level, RIR files, etc.
%       2.2 randomly generate pairs (set) of samples for pairwise (setwise)
%       training, such as deep metric learning, deep clustering, deep LDA,
%       etc.
% Created: 21 Apr 2016
% Last modified: 21 Apr 2016
% Author: Xiong Xiao, Nanyang Technological University, Singapore.
%
function [block_data] = BlockDataGeneration(data, para, sentIdxInBlock)


for si = 1:length(data)
    block_data(si).data = data(si).data(sentIdxInBlock);
    if isfield(data(si), 'base')
        block_data(si).base = data(si).base;
    end
end


end
