function rbm = train_RBM_CD_v2(init_rbm, visible, visible_cv, para)

% Decide the block size
max_block_size = decide_max_block_size_RBM(para);
nSent = length(visible);
for i=1:nSent
    if isfield(para, 'inputFeature') && para.inputFeature == 0
        if isfield(para, 'avgFramePerUtt') && para.avgFramePerUtt >0
            frameInSent(i) = para.avgFramePerUtt;
        else
            frameInSent(i) = 1000;  % Assume 700 frames per utterance
        end
    else
        frameInSent(i) = size(visible{i},2);
    end
end
avgFramePerSent = sum(frameInSent)/nSent;
block_size = 500;
if block_size*avgFramePerSent > max_block_size
    block_size = max(1,floor(max_block_size / avgFramePerSent));
end
nBlock = ceil(nSent/double(block_size));

% Assign sentences to blocks
visibleInBlock = {}; targetInBlock = {};
randomSentIdx = randperm(nSent);
actual_block_size = [];
for i=1:nBlock
    actual_block_size(i) = min(block_size, nSent-sum(actual_block_size(1:i-1)));
    sentIdxInBlock{i} = randomSentIdx( sum(actual_block_size(1:i-1))+1 : sum(actual_block_size(1:i)) );
    randFrameIdxInBlock{i} = randperm(sum(frameInSent(sentIdxInBlock{i})));     % note that this random frame index is not correct if input is file list rather than feature list. 
end

numdims = para.layerSize(1);
numhid = para.layerSize(2);

% -------------------- Initialize weights ----------------------- %
if length(init_rbm) == 0
    % Initializing symmetric weights and biases.
    rbm.vishid     = 0.1*randn(numdims, numhid);    % weight between visible and hidden units
    rbm.hidbiases  = zeros(1,numhid);               % biases of hidden units
    rbm.visbiases  = zeros(1,numdims);
    %rbm.visbiases  = ones(1,numdims);              % biases of visible units, may use negative biases between -3.9 to -4.1
else
    rbm = init_rbm;
end

% -------------- set the learning hyper-parameters ---------------- %

% Separate data into mini-batches
if isfield(para, 'batchSize');     batchSize = para.batchSize;
else     batchSize = 256;  end
% set the maximum allowed number of iterations
if isfield(para, 'maxItr');    maxIter = para.maxItr;
else    maxIter = 300; end
% set minimum number of iterations
if isfield(para,'minItr');    minIter = para.minItr;
else    minIter= 1; end
% set learning rate
epsilonw      = para.learning_rate;   % Learning rate for weights
epsilonvb     = para.learning_rate;   % Learning rate for biases of visible units
epsilonhb     = para.learning_rate;   % Learning rate for biases of hidden units
if isfield(para, 'displayInterval');    displayInterval = para.displayInterval;
else    displayInterval = 300;  end
if isfield(para, 'actual_LR')==0
    para.actual_LR = [];
end

% initialize momentum
vishidinc  = 0; hidbiasinc = 0; visbiasinc = 0;

% --------------- inititialize monitoring variables --------------- %
cost = [];

% --------------- START THE TRAINING --------------- %
if isfield(para, 'start_learning_rate_reduction')
    start_learning_rate_reduction = para.start_learning_rate_reduction;
else
    start_learning_rate_reduction = 0;
end

nFr_seen = 0;   
nFr_reduce_lr = 5e5;
next_Milestone = nFr_reduce_lr*1;

for itr = 1:maxIter
    pause(1);
    
    cost_train = [];
    if para.useGPU; 	cost_train= gpuArray(cost_train);        end
    cost_pure_train = [];
    if para.useGPU; 	cost_pure_train= gpuArray(cost_pure_train);        end
    
    for blk_i = 1:nBlock
        if nFr_seen > next_Milestone
            nHour = round(nFr_seen/36e4);
            lr_scale = max(0.2, 1- nHour/100 );
%             epsilonw      = para.learning_rate * lr_scale;   % Learning rate for weights
%             epsilonvb     = para.learning_rate * lr_scale;   % Learning rate for biases of visible units
%             epsilonhb     = para.learning_rate * lr_scale;   % Learning rate for biases of hidden units
            epsilonw      = epsilonw * 0.98;   % Learning rate for weights
            epsilonvb     = epsilonvb * 0.98;   % Learning rate for biases of visible units
            epsilonhb     = epsilonhb * 0.98;   % Learning rate for biases of hidden units
            fprintf('Trained with %d hours of data, reducing learning rate by %f to %f\n', nHour, lr_scale, epsilonw);
            next_Milestone = next_Milestone + nFr_reduce_lr;
        end
        
        if para.useGPU && isfield(para, 'displayGPUstatus') && para.displayGPUstatus==1
            gpuStatus = gpuDevice;
            fprintf('GPU has %2.2E free memory!\n', gpuStatus.FreeMemory);
        end

        % ------------- prepare data for current block ------------- %
        
        if isfield(para, 'sequential')==0 || para.sequential == 0
            para.sentenceMinibatch = 0;
        else
            para.sentenceMinibatch = 1;     % each sentence is a minibatch
        end
        aux_data_tmp = [];
        [minibatch] = MinibatchPackaging_v2(visible(sentIdxInBlock{blk_i}), [], aux_data_tmp, para);

        % ----------- train the network on current block ------------- %

        cost_tmp = zeros(minibatch.nBatch,1);
        if para.useGPU; 	cost_tmp= gpuArray(cost_tmp);        end
        cost_pure_tmp = zeros(minibatch.nBatch,1);
        if para.useGPU; 	cost_pure_tmp= gpuArray(cost_pure_tmp);        end
        
        for batch_i=1:minibatch.nBatch
            
            data = minibatch.visible(:,minibatch.idx1(batch_i):minibatch.idx2(batch_i))';
            nFr_seen = nFr_seen + size(data,1);

            [cost_tmp(batch_i), grad, cost_pure_tmp(batch_i), negdata] = RBM_cost(data, rbm, para);
            
            % ANTI-WEIGHT-EXPLOSION PROTECTION (Gaussian-Bernoulli RBM) (from KALDI)
            
            % in the following section we detect that the weights in Gaussian-Bernoulli RBM
            % are about to explode. The weight explosion is caused by large variance of the
            % reconstructed data, which causes a feed-back loop that keeps increasing the weights.
            %
            % To avoid explosion, the variance of the visible-data and reconstructed-data
            % should be about the same. The model is particularly sensitive at the very
            % beginning of the CD-1 training.
            %
            % We compute variance of a)input mini-batch b)reconstruction.
            % When the ratio b)/a) is larger than 2, we:
            % 1. scale down the weights and biases by b)/a) (for next mini-batch b)/a) gets 1.0)
            % 2. shrink learning rate by 0.9x
            % 3. reset the momentum buffer
            %
            % Also a warning message is put to log. Note that in later stage
            % the learning-rate returns to its original value.
            %
            % An alternative approach is to use smaller values in weight-matrix initialization.
            
            if strcmpi(para.InputPDF, 'Gaussian')  % use Gaussian distribution for the inputs
                pos_vis_std = std(mat2vec(data));
                neg_vis_std = std(mat2vec(negdata));
                
                if pos_vis_std * 2 < neg_vis_std
                    % scale-down the weights and biases
                    scale = pos_vis_std / neg_vis_std;
                    rbm.vishid = rbm.vishid * scale;
                    rbm.visbiases = rbm.visbiases * scale;
                    rbm.hidbiases = rbm.hidbiases * scale;
                    
                    % reduce the learning rate
                    epsilonw = epsilonw * 0.9;
                    epsilonvb = epsilonvb * 0.9;
                    epsilonhb = epsilonhb * 0.9;
                    
                    % clear the momentum memory
                    vishidinc = 0;
                    visbiasinc = 0;
                    hidbiasinc = 0;
                    
                    fprintf('Mismatch between pos_vis and neg_vis variances, danger of weight explosion. a) Reducing weights with scale %f; b) Lowering learning rate to %f; [pos_vis_std: %f, neg_vis_std: %f]\n', scale, epsilonw, pos_vis_std, neg_vis_std);
                    continue;
                end
            end
            
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if max(max(abs(grad.vishid)))>10
                continue;
            end
            
            momentum = para.momentum(min(itr,length(para.momentum)));
            vishidinc = momentum*vishidinc + epsilonw * grad.vishid;
            visbiasinc = momentum*visbiasinc + epsilonvb * grad.visbiases;
            hidbiasinc = momentum*hidbiasinc + epsilonhb * grad.hidbiases;
        
            if max(max(abs(vishidinc)))>10
                vishidinc = 0; %max(-10, vishidinc);
                vishidinc = 0; %min(vishidinc, 10);
            end
            
            rbm.vishid = rbm.vishid + vishidinc;
            rbm.visbiases = rbm.visbiases + visbiasinc;
            rbm.hidbiases = rbm.hidbiases + hidbiasinc;
                        
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if mod(batch_i,displayInterval)==0
                fprintf('Cost at iteration %i, block %i of %d, minibatch %d of %d = %2.2f/%2.2f, weight norm = %2.2f - %s\n', ...
                    itr, blk_i, nBlock, batch_i, minibatch.nBatch, mean(cost_tmp(batch_i-displayInterval+1 : batch_i)), ...
                     mean(cost_pure_tmp(batch_i-displayInterval+1 : batch_i)), sum(sum(rbm.vishid.^2)), datestr(now));
                pospath = reshape(data(1,:),size(data,2)/para.context,para.context);
                negpath = reshape(negdata(1,:),size(data,2)/para.context,para.context);
                imagesc([pospath zeros(size(pospath,1),3) negpath zeros(size(pospath,1),3) (pospath-negpath)/2]); colorbar
                pause(0.1);
            end
        end
        cost_train = [cost_train; cost_tmp];
        cost_pure_train = [cost_pure_train; cost_pure_tmp];
        clear minibatch;
    end
    cost(itr) = gather(mean(cost_train));    
    cost_pure(itr) = gather(mean(cost_pure_train));    
    
    if isinf(cost(itr))
        pause(0.1);
    end
    
    % ------------------ save current network ----------------------- %

    if isfield(para, 'saveModelEveryXIter') && mod(itr,para.saveModelEveryXIter)==0
        if para.useGPU
            rbm.vishid = gather(rbm.vishid);
            rbm.visbiases = gather(rbm.visbiases);
            rbm.hidbiases = gather(rbm.hidbiases);
        end
        save([para.output sprintf('.itr%d.L2_%2.1E.LR%2.2E.Cost%2.2f.mat', itr, para.L2weight, epsilonw, cost_pure(end))], ...
            'rbm', 'cost', 'cost_pure', 'para');
    end
    
    % ------------- stopping criterion and learn rate reduction --------- %

    if itr>1 || start_learning_rate_reduction==1
        if itr>=minIter && (cost(end-1)-cost(end))/cost(end) <para.stopImprovement/100
            fprintf('Improvement is less than %2.3f%%, stop the training!\n', para.stopImprovement);
            break;
        end
        if start_learning_rate_reduction ==1 || (cost(end-1)-cost(end))/cost(end) <para.reduceLearnRate/100
            start_learning_rate_reduction=1;
            epsilonw = epsilonw * para.reduceLearnRateSpeed;
            epsilonvb = epsilonvb * para.reduceLearnRateSpeed;
            epsilonhb = epsilonhb * para.reduceLearnRateSpeed;
            fprintf('Cost reduction less than %2.3f%%, learning rate reduced by %2.2f%% to %f\n', para.reduceLearnRate, 100-100*para.reduceLearnRateSpeed, epsilonw);
        end
    end 
end
end
