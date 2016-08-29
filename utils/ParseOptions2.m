% Set the parameters not set by user to the default values. Also can print
% out the detailed option list. 
% Xiong Xiao, Nanyang Technological University, Singapore
% Last modified: 05 Nov 2015. 
%
function para = ParseOptions2(para, displayDefaultValues)
DisplayHelpInfoOnly = 0;
if nargin<1
    DisplayHelpInfoOnly = 1;
end
if nargin<2
    displayDefaultValues = 0;
end

% Define a data structure that contains the
configGeneral = {'useGPU',     0, 	'whether to use GPU for computation. The other option is CPU.';
    'storeDataInCell',  0, 'if true, store data in cell array, and copy data to GPU in minibatch. If false, store data in matrix and copy data to GPU in block.';
    'DEBUG',            0, 'whether to display network debug information.';
    'singlePrecision',  1,  'whether to use single precision computation. For gradient check, need to use double precision.';
    'maxItr',           100, 'the maximum number of iterations for DNN training.';
    'minItr',           10,  'the minimum number of iterations for DNN training.';
    'displayInterval',  100, 'display the training process for every X number of minibatches.';
    'skipInitialEval',  0,  'whether to skip the evaluation on cross validation data before network training.';
    'displayGPUstatus', 0, 'whether to display GPU status after every block of training samples.';
    'checkGradient',    0, 'whether to check gradient computation is correct.';
    'saveModelEveryXIter', 1, 'for every X number of iterations, the network will be saved.';
    'saveModelEveryXhours', 0, 'for every X hours of training data (360000 time steps an hour following MFCC features, the network will be saved.';
    'cost_func',        [], 'the structure that defines the cost function information.';
    'preprocessing',    [], 'a cell array that defines the preprocessing of input streams. Each cell itself is a cell array that defines the steps of the processing for the corresponding stream.';
    'output',           '', 'the file name prefix (including path) of the output model.';
    'storeDataInCell', 0, 'whether to store minibatch in cell or in matrix.';
    'displayTag',       '', 'anything you want to see on the progress report. e.g. some name to remind you what is running.';
    };

configNet = {'sequential',       0, 'whether to use sequential training.';
    'nSequencePerMinibatch', 1, 'defines the number of concurrent sequences in each minibatch, if sequential is true. Usually for CNN or LSTM training.';
    'variableLengthMinibatch', 0, 'whether the trajectories in a minibatch have the same length.';
    'maxNumSentInBlock', 1000, 'maximum number of sentences to read in for a block. ';
    'sentenceMinibatch', 0, 'whether to use sentence as minibatch. This is true if sequential training is used.';
    'randomizedBlock',  1,  'whether to randomize the sentences to form the blocks.';
    'useNegbiasInit',   0,  'whether to use negative bias initialization.';
    'useGaussInit',     0,  'whether to use Gaussian distributed random numbers to initialize network parameters. The other option is use uniform distribution.';
    'batchSize',        256, 'the size of minibatch in terms of training samples. This configuration is not used if sequential training is used.';
    'learningScheme', 'decayIfNoImprovement', 'scheme to adjust learning rate: 1) decayIfNoImprovement; 2) expDecay.';
    'learning_rate',    1e-4, 'the global learning rate.';
    'learning_rate_floor', 0, 'the minimum learning rate';
    'start_learning_rate_reduction', 0, 'whether to start decaying learning rate every iteration.';
    'reduceLearnRate',  0.5, 'if the cost function is reduced by less than this amount in percentage between two iterations, learning rate will start to reduce.';
    'reduceLearnRateSpeed', 0.5, 'the factor multiplied to learning rate at the end of every iteration once learning rate start to reduce.';
    'stopImprovement',  0.1, 'if the cost function is reduced by less than this amount in percentage between two iterations, the network training will terminate.';
    'learning_rate_decay_rate', 0.995, 'the rate of decaying learning rate after every half an hour of data, which is equal to 180000 training samples.';
    'momentum',         0, 'the value of momentum used in network update. If an array is provided, the actual momentum will be momentum(min(length(momentum),itr), where itr is the iteration number.';
    'rmsprop_decay',    0, 'the rate of rmsprop decay.';
    'rmsprop_damping',  1e-2, 'the rate of rmsprop damping rate.';
    'L1',               1e-1, 'the target of sparsity regularization, i.e. the expected percentage of hidden nodes that is activated'.';
    'L1weight',         0, 'the weight of L1 sparsity regularization.';
    'L2weight',         0, 'the weight of L2 regularization, i.e. weight decay.';
    'WeightTyingSet',   [], 'the set of layers whose weights are tied. This is a cell array of layer indexes. In each cell, the layers indexed share the same weights.';
    'gradientClipThreshold', 0, 'if the absolute valude of the gradient of a parameter is larger than this threshold, it will be set to this value, with sign unchanged. Disabled if set to 0.';
    'weight_clip',      0, 'if the absolute value of a weight is larger than this threshold, clip the weight value. Disabled if it is set to 0.';
    'restore2prevModelIfFail', 1, 'if the cross validation cost increases after one iteration of training, restore the network to the previous version.';
    'displayLBFGS', 0, 'whether to display LBFGS progress.';
    };

configIO = {'mode', 'normal', 'input modes: 1) normal; 2) dynamicPair; 3) dynamicDistortion.';
    'baseType', 'matrix', 'the format of the base: 1) matrix; 2) tensor; 3) cell.';
    'nStream', 2, 'the number of input and output streams. E.g. if there is one input stream and one target for conventional supervised training, nStream=2.';  % nStream must be put as the first config in configIO.
    'asynchronous', 0, 'whether the input streams are asynchronous. If not, the frame rates of the streams will be normalized to the maximum frame rates among them, and the number of frames in each sentence will be set to the minimum among the streams.';
    'DataSyncSet', [], 'a cell array, each cell contains an array of input stream index that are going to be synchronized.';
    'ApplyVADSet', [], 'a cell array, each cell contains an array of input stream index, where the first index is the stream to be VADed, and the second stream is the VAD stream.';
    'vadAction',   [], 'a cell array of actions for ApplyVADSet. E.g.: 1) "concatenation"; 2) "segmentation A B" which divide speech segments into length of A with shift B.';
    'isTensor',     0, 'whether the stream will be a 3D tensor, usually used to represent a group of 2D trajectories.'; 
    'avgFramePerUtt',   1000,  'average number of frames per utterance. Used when inputs are filenames rather than features.';
    'inputFeature', 1, 'an array of binary indicators specifying the format of the input streams. The length of the array equals the number of input streams. 0: file names, 1: features.';
    'fileReader',   [], 'an array of file reader structures that defines how to read the features if inputFeature is 0.';
    'isIndex',      0, 'an array of binary indicators specifying whether the value in input data is an index in the base database.';
    'sparse',       0, 'an array of binary indicators specifying whether the input stream is sparse data.';
    'context',      1, 'an array of integer specifying the size of context frames used as the input of the network. Only applicable when input is a trajectory.';
    'context_skip', 0, 'when context is larger than 1, context_skip defines how many frames are skipped when taking the context frames. E.g. if it is 1, then for every 2 contextual frames we take one.';
    'frame_rate',   100, 'the number of frames per second.';
    'ClassLabel4EvenBlock', 0, 'the stream ID that contains the class label information for generating blocks covers as diverse classes as possible. This option is disabled by default.';
    'ClassLabel4EvenBlock_refill', 0, 'whether to refill a minority class after its samples are used up.';
    'shuffleByDurationStreamIdx', 0, 'set to positive stream index if we want to shuffle the data based on duration. Useful for variable length multi-sentence minibatch.\n';
    'blockSizeMultiplier', 1, 'multiply the block size by a number.';
    };

if DisplayHelpInfoOnly
    fprintf('General Configurations:\n');    DisplayConfig(configGeneral);
    fprintf('Configurations for Input/Output:\n');    DisplayConfig(configIO);
    fprintf('Configurations for Network:\n');    DisplayConfig(configNet);
%     fprintf('Configurations for Training:\n');    DisplayConfig(configTrain);
    return;
end

% Set unset parameters to their default values
para = SetConfig(para, configGeneral, displayDefaultValues);
para.IO = SetConfig(para.IO, configIO, displayDefaultValues);
para.NET = SetConfig(para.NET, configNet, displayDefaultValues);

% Verify the configurations such that there is no conflicting
% configurations. 
fprintf('\n');

if para.checkGradient && para.singlePrecision
    fprintf('Warning: gradient checking should use double precision, your setting is single precision! Change to double precision\n');
    para.singlePrecision = 0;
end

must_set_options = {'cost_func', 'output'};
for i=1:length(must_set_options)
    if isempty(para.(must_set_options{i}))
        fprintf('Error: %s must be not empty\n', must_set_options{i});
    end
end

for i=1:length(para.IO.inputFeature)
    if para.IO.inputFeature(i)==0 
        if length(para.IO.fileReader)<i || isfield(para.IO.fileReader(i), 'name')==0 || length(para.IO.fileReader(i).name) == 0
            fprintf('Error: input stream %d is file names, but its fileReader is not defined.\n', i);
            return;
        end
    end
end

end

%% Display the detailed information about the configurations
function DisplayConfig(config)
for i=1:size(config,1)
    fprintf('  %s --- %s Default value is ', config{i,1}, config{i,3});
    default_value = config{i,2};
    if isempty(default_value)
        fprintf('empty.\n');
    elseif ischar(default_value)
        fprintf('%s.\n', default_value);
    elseif default_value == round(default_value)   % is an integer
        fprintf('%d.\n', default_value);
    elseif isnumeric(default_value)
        fprintf('%f\n', default_value);
    else
        fprintf('%s\n', default_value);
    end
end
fprintf('\n');
end

%% Set unset parameters to their default values
function para = SetConfig(para, config, displayDefaultValues)
% the following configurations needs to be set for every input stream. 
ArrayConfig = {'inputFeature', 'fileReader', 'isIndex', 'sparse', 'context', 'context_skip', 'frame_rate', 'isTensor'};

for i=1:size(config,1)
    config_name = config{i,1};
    default_value = config{i,2};
    
    isArrayConfig = strcmpi(ArrayConfig, config_name);
    if sum(isArrayConfig)>0
        if isfield(para, config_name)==0
            if strcmpi(config_name, 'fileReader')
                para.(config_name)= repmat(struct('name', ''), 1, para.nStream);
            else
                para.(config_name) = repmat(default_value, 1, para.nStream);
            end
        elseif isfield(para, config_name) && length(para.(config_name))~=para.nStream
            fprintf('Error: the number of elements in %s (%d) is not the same as the number of input streams (%d)\n', length(para.(config_name)), para.nStream);
        end
    elseif isfield(para, config_name)==0
        para.(config_name) = default_value;
        if displayDefaultValues
            fprintf('%s set to default value: ', config_name);
            if isempty(default_value)
                fprintf('[].\n');
            elseif default_value == round(default_value)   % is an integer
                fprintf('%d.\n', default_value);
            elseif isnumeric(default_value)
                fprintf('%f\n', default_value);
            else
                fprintf('%s\n', default_value);
            end
        end
    end
end
end
