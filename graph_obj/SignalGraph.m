classdef SignalGraph
    properties
        % configuration
        
        % data
        layer={};
        WeightUpdateOrder = [];
    end
    methods
        function obj = signalGraph(layer)
            obj.layer = layer;
        end
        
        
        
        function obj = initialize(obj, para)
            layer = obj.layer;
            negbias = para.NET.useNegbiasInit;
            gauss = para.NET.useGaussInit;
            
            % Initialize parameters randomly based on layer sizes.
            nNode = 0;
            for i=1:length(layer)
                if strcmpi(layer{i}.name, 'affine') || strcmpi(layer{i}.name, 'input') || strcmpi(layer{i}.name, 'word2vec')
                    nNode = nNode + layer{i}.dim(1);
                end
            end
            
            r  = sqrt(6) / sqrt(nNode);   % we'll choose weights uniformly from the interval [-r, r]
            for i=1:length(layer)
                if strcmpi(layer{i}.name, 'affine') || strcmpi(layer{i}.name, 'tconv')
                    if layer{i}.update && para.NET.rmsprop_decay>0
                        layer{i}.gradW_avg_square = 0;
                    end
                    if length(layer{i}.prev)>1; continue; end   % if there are more than 1 input layer, one of them will provide parameters
                    if isfield(layer{i}, 'W')==0 || sum(abs(size(layer{i}.W)-layer{i}.dim))>0
                        if gauss
                            layer{i}.W = 3/sqrt(layer{i}.dim(2)) * randn(layer{i}.dim);
                        else
                            layer{i}.W = rand(layer{i}.dim) * 2 * r - r;
                        end
                    end
                    if layer{i}.update && para.NET.rmsprop_decay>0
                        layer{i}.gradb_avg_square = 0;
                    end
                    if isfield(layer{i}, 'b')==0 || sum(abs(size(layer{i}.b,1)-layer{i}.dim(1)))>0
                        if negbias
                            layer{i}.b = rand(layer{i}.dim(1),1)/5 - 4.1;
                        else
                            layer{i}.b = zeros(layer{i}.dim(1),1);
                        end
                    end
                elseif strcmpi(layer{i}.name, 'word2vec')
                    trans_size = layer{i}.dim / layer{i}.context;
                    if isfield(layer{i}, 'W')==0 || sum(abs(size(layer{i}.W)-trans_size))>0
                        if gauss
                            layer{i}.W = 3/sqrt(trans_size(2)) * randn(trans_size);
                        else
                            layer{i}.W = rand(trans_size) * 2 * r - r;
                        end
                    end
                elseif strcmpi(layer{i}.name, 'LSTM')
                    input_size = layer{i}.dim(2); hidden_size = layer{i}.dim(1);
                    input_and_recurrent_size = hidden_size*(1+layer{i}.usePastState)+input_size;
                    if isfield(layer{i}, 'W')==0 || sum(abs(size(layer{i}.W)-[hidden_size*4 input_and_recurrent_size]))>0
                        % the weight matrix of LSTM is organized as follows:
                        % W = [ W_cf W_hf W_xf;
                        %       W_cc W_hc W_xc;
                        %       W_ci W_hi W_xi;
                        %       W_co W_ho W_xo;]
                        % where W_cf is the weights that connects past cell states to
                        % forget gates, W_hc is the weights between past hidden activation
                        % to candidate states, W_xi is the weights beween input features to
                        % input gates. Other weights are similarly defined.
                        layer{i}.W = randn(4 * hidden_size, input_and_recurrent_size) / sqrt(input_and_recurrent_size);
                        if layer{i}.usePastState
                            % we need to make W_cc zero as there is no connection between past
                            % cell states to candidate cell states.
                            layer{i}.W( hidden_size+1:hidden_size*2, 1:hidden_size ) = 0;
                        end
                    end
                    if isfield(layer{i}, 'b')==0 || sum(abs(size(layer{i}.b,1)-hidden_size*4))>0
                        layer{i}.b = zeros(hidden_size*4,1);
                    end
                end
            end
            
            if ~isempty(para.NET.WeightTyingSet)
                for i=1:length(para.NET.WeightTyingSet)
                    currTyingSet = para.NET.WeightTyingSet{i};
                    [dimMismatch, isTranspose] = VerifyTiedLayers(layer(currTyingSet));
                    
                    baseNode = layer{currTyingSet(1)};
                    for j=2:length(currTyingSet)
                        if isfield(baseNode, 'W')
                            if isTranspose(j)
                                layer{currTyingSet(j)}.W = baseNode.W';
                            else
                                layer{currTyingSet(j)}.W = baseNode.W;
                            end
                        end
                        if isfield(baseNode, 'b')   && ~isTranspose(j)  % if the two shared layers are transpose of each other, we don't share the bias
                            layer{currTyingSet(j)}.b = baseNode.b;
                        end
                    end
                end
            end
            obj.layer = layer;
        end
        
        
        
        
        function [cost_func_tmp, obj] = forwardBackward(obj, batch_data, para, option)
            % This function does both the forward pass and backward pass of DNN.
            % The forward pass is like a multilayer nonlinear transform of the input,
            % while the backward pass computes the gradients.
            % The inputs of the function are:
            %   theta - a 1-D array which contains all the parameers of the DNN
            %   visible - the input of the DNN, a d-by-m matrix, where d is the
            %       dimension and m is the number of samples
            %   target - the desired output of the DNN, a n-by-m matrix, where n is the
            %       number of targets and m is the number of sampels
            %   cost_scale - a matrix of the same size as target. it is used to enable
            %       Mahalonobis distance when cost function is mean square error
            %   para - a structure that contains various settings of the DNN
            %   mode - 1,2,3,or 4. See the below for details
            % The outputs of the function are
            %   cost - the value of the cost function given the inputs
            %   grad - the gradient of the parameters
            %   output - the output of the DNN given the input
            %
            % Author: Xiong Xiao, NTU
            % Date Created: 10 Oct 2013
            % Last Modified: 24 Jul 2015
            %
            % mode = 1; run both forward and backward pass
            % mode = 2; only run forward pass to generate network output
            if para.singlePrecision==0;    precision = 'double';
            else     precision = 'single'; end
            
            nLayer = length(layer);
            % Run forward propogation
            for i=1:nLayer
                if isfield(layer{i}, 'prev');   prev_layers = layer(i+layer{i}.prev);    end
                switch lower(layer{i}.name)
                    case 'ignore'
                        % just pass
                    case 'input'
                        layer{i}.a = data{layer{i}.inputIdx};
                    case 'idx2vec'
                        [layer{i}.a, layer{i}.validFrameMask] = F_idx2vec(prev_layers{1}, layer{i}, para.singlePrecision);       % do not support variable length yet
                    case 'affine'
                        input_layer_idx = length(prev_layers);
                        if strcmpi(prev_layers{input_layer_idx}.name, 'input') && para.IO.sparse(prev_layers{input_layer_idx}.inputIdx)
                            [layer{i}.a, layer{i}.validFrameMask] = F_sparse_affine_transform(prev_layers{1}, layer{i}.W, layer{i}.b, para.singlePrecision);
                        else
                            if issparse(prev_layers{input_layer_idx}.a);  prev_layers{input_layer_idx}.a = full(prev_layers{input_layer_idx}.a); end
                            [layer{i}.a, layer{i}.validFrameMask] = F_affine_transform(prev_layers, layer{i});
                        end
                    case 'add'
                        layer{i}.a = F_add(prev_layers);
                    case 'matrix_multiply'
                        [layer{i}.a, layer{i}.validFrameMask] = F_matrix_multiply(prev_layers, layer{i});
                    case 'hadamard'
                        [layer{i}.a, layer{i}.validFrameMask] = F_hadamard(prev_layers);
                    case 'word2vec'
                        layer{i}.a = F_word2vec(prev_layers{1}.a, layer{i}.W, para.singlePrecision);    % do not support variable length yet
                    case 'concatenate'
                        layer{i}.a = F_concatenate(prev_layers);    % do not support variable length yet
                    case 'extractdims'
                        layer{i}.a = F_ExtractDims(prev_layers{1}, layer{i}.dimIndex);
                    case 'reshape'
                        layer{i}.a = F_reshape(prev_layers{1}, layer{i});
                    case 'repmat'
                        layer{i}.a = F_repmat(prev_layers{1}, layer{i});
                    case 'transpose'
                        layer{i}.a = F_transpose(prev_layers{1}, layer{i});
                    case 'permute'
                        layer{i}.a = F_permute(prev_layers{1}, layer{i});
                    case 'frame_select'
                        [layer{i}.a, layer{i}.validFrameMask] = F_frame_select(prev_layers{1}, layer{i});
                    case 'frame_shift'
                        [layer{i}.a, layer{i}.validFrameMask] = F_frame_shift(prev_layers{1}, layer{i});
                    case 'copyvec2mat'
                        [layer{i}.a] = F_copyVec2Mat(prev_layers{1}, layer{i});
                    case 'weighting'
                        layer{i}.a = F_weighting(prev_layers{1}.a, layer{i}.W, layer{i}.b);             % do not support variable length yet
                    case 'cmn'
                        [layer{i}.a, layer{i}.validFrameMask] = F_cmn(prev_layers{1});
                    case 'absmax_norm'
                        [layer{i}.a, layer{i}.validFrameMask] = F_absmax_norm(prev_layers{1}, layer{i});
                    case 'minmax_norm'
                        [layer{i}.a, layer{i}.validFrameMask] = F_minmax_norm(prev_layers{1}, layer{i});
                    case 'linear'
                        layer{i}.a = prev_layers{1}.a;
                    case {'sigmoid'}
                        layer{i}.a = F_sigmoid(prev_layers{1});
                    case {'exp'}
                        layer{i}.a = F_exp(prev_layers{1});
                    case {'mu_law'}
                        layer{i}.a = F_mu_law(prev_layers{1}, layer{i});
                    case {'tanh'}
                        layer{i}.a = F_tanh(prev_layers{1});
                    case 'softmax'
                        layer{i}.a = F_softmax(prev_layers{1});
                    case 'multi_softmax'
                        layer{i}.a = F_multi_softmax(prev_layers{1}, layer{i}.TaskVocabSizes);
                    case 'logistic'
                        [layer{i}.a, layer{i}.acc] = F_logistic(prev_layers, layer{i});
                    case 'cosine'
                        layer{i}.a = F_cosine(prev_layers);
                    case 'inner_product'
                        layer{i}.a = F_inner_product(prev_layers);
                    case 'inner_product_normalized'
                        layer{i}.a = F_inner_product_normalized(prev_layers);
                    case 'relu'
                        layer{i}.a = F_relu(prev_layers{1}, layer{i});
                    case 'maxout'
                        
                    case 'largerthan'
                        layer{i}.a = F_largerThan(prev_layers{1},layer{i});
                        
                    case 'mean'
                        layer{i}.a = F_mean(prev_layers{1}, layer{i});
                    case 'median'
                        layer{i}.a = F_median(prev_layers{1}, layer{i});
                    case 'max'
                        layer{i}.a = F_max(prev_layers{1}, layer{i});
                    case 'min'
                        layer{i}.a = F_min(prev_layers{1}, layer{i});
                        
                    case 'weight2activation'
                        layer{i} = F_weight2activation(layer{i});
                        
                    case 'tconv'
                        [layer{i}.a, layer{i}.X2] = F_tconv(prev_layers, layer{i});
                    case 'tmaxpool'
                        [layer{i}.a, layer{i}.idx, layer{i}.validFrameMask] = F_tmaxpool(prev_layers{1}, layer{i});
                        
                    case 'weighted_average'
                        [layer{i}.a,layer{i}.weights] = F_weighted_average(prev_layers);
                        
                    case 'delta'
                        [layer{i}.a, layer{i}.validFrameMask] = F_dynamic_feat(prev_layers{1});
                    case 'log'
                        [layer{i}.a, layer{i}.validFrameMask] = F_log(prev_layers{1}, layer{i}.const);
                    case 'power'
                        layer{i}.a = F_power_spectrum(prev_layers{1});
                    case 'sqrt'
                        layer{i}.a = sqrt(prev_layers{1}.a);
                    case 'splice'
                        [layer{i}.a, layer{i}.validFrameMask] = F_splice(prev_layers{1}, layer{i}.context);
                    case 'mel'
                        [layer{i}.a, layer{i}.validFrameMask] = F_affine_transform(prev_layers, layer{i});
                    case 'power_split'
                        layer{i}.a = F_power_spectrum_split(prev_layers{1}.a);
                    case 'beamforming'
                        [layer{i}.a] = F_beamforming(prev_layers, layer{i});
                    case 'beamforming_freeweight'
                        layer{i}.a = F_beamforming_freeWeight(prev_layers{1}, layer{i});
                    case 'beamforming_gaintdoa'
                        layer{i}.a = F_beamforming_gainTDOA(prev_layers{1}, layer{i});
                    case 'filter'
                        layer{i}.a = F_filter(prev_layers);
                    case 'comp_gcc'
                        [layer{i}.a, layer{i}.validFrameMask] = F_comp_gcc(prev_layers{1}, layer{i});
                    case 'stft'
                        if para.checkGradient; layer{i}.doDithering=0; end
                        [layer{i}.a, layer{i}.validFrameMask] = F_stft(prev_layers{1}, layer{i});
                        
                    case 'spatialcovmask'
                        layer{i}.a = F_SpatialCovMask(prev_layers, layer{i});       % do not support variable length yet
                    case 'spatialcovsplitmask'
                        layer{i}.a = F_SpatialCovSplitMask(prev_layers, layer{i});       % do not support variable length yet
                    case 'mvdr_spatialcov'
                        layer{i} = F_MVDR_spatialCov(prev_layers{1}, layer{i});       % do not support variable length yet
                        
                    case 'cov'
                        layer{i}.a = F_cov(prev_layers{1}.a);       % do not support variable length yet
                    case 'logdet'
                        layer{i}.a = F_logdet(prev_layers{1}.a);    % do not support variable length yet
                    case 'll_gmm'
                        layer{i} = F_ll_gmm(prev_layers{1}.a, layer{i});    % do not support variable length yet
                    case 'll_gaussian'
                        layer{i}.a = F_ll_gaussian(prev_layers, layer{i});    % do not support variable length yet
                        
                    case 'tdoa2weight'
                        layer{i}.a = F_tdoa2weight(prev_layers{1}.a, layer{i}.freqBin);
                    case 'real_imag2bfweight'
                        if isfield(layer{i}, 'online')==0; layer{i}.online = 0; end
                        [layer{i}.a, layer{i}.validFrameMask] = F_real_imag2BFweight(prev_layers{1}, layer{i}.freqBin, layer{i}.online);
                    case 'realimag2complex'
                        layer{i}.a = F_realImag2complex(prev_layers{1});
                    case 'complex2realimag'
                        layer{i}.a = F_complex2realImag(prev_layers{1});
                    case 'mse'
                        layer{i}.a = F_mean_square_error(prev_layers, layer{i});
                    case 'mixture_mse'
                        layer{i} = F_mixture_mse(prev_layers, layer{i});
                    case 'jointcost'
                        layer{i}.a = F_jointCost(prev_layers{1}, layer{i});
                    case 'cross_entropy'
                        [layer{i}.a, layer{i}.acc] = F_cross_entropy(prev_layers, layer{i});
                    case 'multi_cross_entropy'
                        [layer{i}.a, layer{i}.acc] = F_multi_cross_entropy(prev_layers, layer{i});
                    case 'lstm'
                        layer{i} = F_LSTM(prev_layers{1}, layer{i});
                    otherwise
                        fprintf('Error: unknown output node type %s!\n', layer{i}.name);
                end
                
                % pass validFrameMask from layer to layer
                if isfield(layer{i}, 'prev') && ~isfield(layer{i}, 'validFrameMask') && ~strcmpi(layer{i}.name, 'ignore')
                    if size(layer{i}.a,3)>1      % for those layers that do not need validFrameMask for computing, we compute the mask here.
                        layer{i}.validFrameMask = getValidFrameMask(prev_layers{1});
                    else    % for two dimensional activations, there is no need to have mask
                        layer{i}.validFrameMask = [];
                    end
                end
                
                if i<nLayer
                    if para.NET.L1weight>0 || (isfield(layer{i}, 'L1weight') && layer{i}.L1weight>0)
                        layer{i}.rho = mean(layer{i}.a,2);
                    end
                end
            end
            
            if mode ==3
                for i=1:length(para.out_layer_idx)
                    tmpOutput = layer{para.out_layer_idx(i)}.a;
                    [~,~,N] = size(tmpOutput);
                    if N==1
                        output{i} = tmpOutput;
                    else
                        currLayer = layer{para.out_layer_idx(i)};
                        if isfield(currLayer, 'validFrameMask')
                            mask = currLayer.validFrameMask;
                            output{i} = PadShortTrajectory(tmpOutput, mask, -1e10);
                        else
                            output{i} = tmpOutput;
                        end
                    end
                end
                cost_func = [];
                return;
            end
            
            % compute the cost function
            nCost = length(para.cost_func.layer_idx);
            if para.useGPU;
                cost_func.subcost = gpuArray.zeros(nCost,1);
                cost_func.subacc = gpuArray.zeros(nCost,1);
            else
                cost_func.subcost = zeros(nCost,1);
                cost_func.subacc = zeros(nCost,1);
            end
            for i=1:nCost
                cost_func.subcost(i) = para.cost_func.layer_weight(i) * layer{para.cost_func.layer_idx(i)}.a;
                if strcmpi(layer{para.cost_func.layer_idx(i)}.name, 'cross_entropy') || strcmpi(layer{para.cost_func.layer_idx(i)}.name, 'logistic') || strcmpi(layer{para.cost_func.layer_idx(i)}.name, 'multi_cross_entropy')
                    cost_func.subacc(i) = layer{para.cost_func.layer_idx(i)}.acc;
                else
                    cost_func.subacc(i) = 0;
                end
            end
            cost_func.cost = sum(cost_func.subcost);
            cost_func.cost_pure = cost_func.cost;
            
            if para.NET.L2weight>0
                L2weight = para.NET.L2weight;
                if para.useGPU
                    L2weight = gpuArray(L2weight);
                end
                for i=2:nLayer
                    if isfield(layer{i}, 'W') && layer{i}.update
                        if isfield(layer{i}, 'W0')
                            tmp = layer{i}.W - layer{i}.W0;   % if we are given a initial weight matrix W0, we measure the difference between W and W0.
                        else
                            tmp = layer{i}.W;   % otherwise, we measure the difference between W and a matrix of zeros.
                        end
                        if isfield(layer{i}, 'mask')        % the mask defines what values can be tuned and what cannot be tuned.
                            tmp = tmp .* layer{i}.mask;
                        end
                        cost_func.cost = cost_func.cost + 0.5* L2weight * sum(sum(tmp.*tmp));
                    end
                end
            end
            
            % if para.NET.L1weight>0    % If use sparsity constraint
            for i=2:nLayer-1
                if isfield(layer{i}, 'rho')
                    % note that we limit the denominator to be greater than a small number
                    if isfield(layer{i}, 'L1')
                        L1 = layer{i}.L1;
                        L1weight = layer{i}.L1weight;
                    else
                        L1 = para.NET.L1;
                        L1weight = para.NET.L1weight;
                    end
                    tmp = L1*log(L1./max(1e-3,layer{i}.rho)) + (1-L1)*log((1-L1)./max(1e-3,1-layer{i}.rho));
                    cost_func.cost = cost_func.cost + L1weight * sum(tmp);
                end
            end
            % end
            
            if para.DEBUG
                hasinf = sum(isinf(cost_func.subcost)) + isinf(cost_func.cost) ;
                hasnan = sum(isnan(cost_func.subcost)) + isnan(cost_func.cost) ;
                if hasnan>0
                    fprintf('NAN detected in cost\n');
                elseif hasinf>0
                    fprintf('Inf detected in cost\n');
                end
            end
            
            if mode==2;
                if 0    % investigate the effect of DNN BF
                    %         imagesc(([layer{end-2}.a layer{end-1}.a]));   pause
                    %         compareUttDbyD(layer{end-2}.a', layer{end-1}.a');
                    spec_noisy = log(abs(layer{end-5}.a(1:257,:)))*2;
                    imagesc([spec_noisy; layer{end-2}.a; layer{end-1}.a; spec_noisy-layer{end-1}.a; layer{end-2}.a-layer{end-1}.a]);
                    hold on;
                    plot(sum( abs(spec_noisy-layer{end-1}.a)/5 ))
                    plot(sum( abs(layer{end-1}.a-layer{end-2}.a)/5 ),'r')
                    legend('MSE-raw', 'MSE-BF');
                    hold off; colorbar
                    
                    pause
                end
                
                return;
            end
            
            % Vectorized implementation of backpropagation
            if para.DEBUG; hasnan=0; end
            for i=nLayer:-1:1
                if isfield(layer{i}, 'prev');   prev_layers = layer(i+layer{i}.prev);    end
                if isfield(layer{i}, 'next');   future_layers = layer(i+layer{i}.next);    end
                if isfield(layer{i}, 'skipBP') && layer{i}.skipBP == 1; continue; end   % some layers do not need to compute gradients, such as comp_gcc and stft
                switch lower(layer{i}.name)
                    case {'input', 'idx2vec', 'enframe', 'comp_gcc', 'stft'} % do nothing
                        
                    case 'frame_select'
                        layer{i}.grad = B_frame_select(prev_layers{1}, future_layers, layer{i});
                    case 'reshape'
                        layer{i}.grad = B_reshape(prev_layers{1}, future_layers, layer{i});
                    case 'repmat'
                        layer{i}.grad = B_repmat(future_layers, layer{i});
                    case 'transpose'
                        layer{i}.grad = B_transpose(future_layers, layer{i});
                    case 'permute'
                        layer{i}.grad = B_permute(future_layers, layer{i});
                        % updatable layers
                    case {'affine', 'mel'}
                        [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_affine_transform(prev_layers, layer{i}, future_layers, i==2);
                    case 'matrix_multiply'
                        [layer{i}.grad] = B_matrix_multiply(prev_layers, layer{i}, future_layers);
                    case 'hadamard'
                        [layer{i}.grad] = B_hadamard(prev_layers, layer{i}, future_layers);
                    case 'word2vec'
                        [layer{i}.grad, layer{i}.grad_W] = B_word2vec(prev_layers, layer{i}, future_layers, para.singlePrecision);
                    case 'copyvec2mat'
                        [layer{i}.grad] = B_copyVec2Mat(prev_layers{1}, layer{i}, future_layers);
                    case {'weighting'}
                        [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_weighting(prev_layers, layer{i}, future_layers);
                    case 'lstm'
                        [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_LSTM(prev_layers{1}, layer{i}, future_layers);
                    case 'add'
                        [layer{i}.grad] = B_add(prev_layers, future_layers, layer{i});
                        
                        % cost layers
                        
                    case 'mse'
                        layer{i}.grad = B_mean_square_error(prev_layers, layer{i});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                    case 'mixture_mse'
                        layer{i}.grad = B_mixture_mse(prev_layers, layer{i});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                    case 'jointcost'
                        layer{i}.grad = B_jointCost(prev_layers{1}, layer{i});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                    case {'multi_cross_entropy', 'cross_entropy'}    %compute the gradient together with softmax
                        % layer{i}.grad = B_cross_entropy(layer(i+layer{i}.prev));
                    case 'logistic'
                        layer{i}.grad = B_logistic(prev_layers, layer{i});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                        
                        % temporal layers: require sequential training
                    case 'mean'
                        layer{i}.grad = B_mean(prev_layers{1}, future_layers, layer{i});
                        
                    case 'tconv'
                        [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b]  = B_tconv(prev_layers, layer{i}, future_layers, i==2);
                    case 'tmaxpool'
                        layer{i}.grad = B_tmaxpool(prev_layers, layer{i}, future_layers);
                    case 'max'
                        layer{i}.grad = B_max(prev_layers, future_layers);
                    case 'weighted_average'
                        [layer{i}.grad, layer{i}.grad_W_raw] = B_weighted_average(prev_layers, layer{i}, future_layers);
                    case 'delta'
                        layer{i}.grad = B_dynamic_feat(layer{i}, future_layers);
                    case 'splice'
                        layer{i}.grad = B_splice(future_layers, layer{i});
                    case 'cmn'
                        layer{i}.grad = B_cmn(future_layers);
                        
                    case 'weight2activation'
                        [layer{i}.grad, layer{i}.grad_W] = B_weight2activation(layer{i}, future_layers);
                        
                        % signal processing layers
                    case 'log'
                        layer{i}.grad = B_log(future_layers, layer{i+layer{i}.prev}.a, layer{i});
                    case 'power'
                        layer{i}.grad = B_power_spectrum(prev_layers{1}.a, future_layers);
                    case 'power_split'
                        layer{i}.grad = B_power_spectrum_split(future_layers, layer{i+layer{i}.prev}.a);
                    case 'filter'
                        layer{i}.grad = B_filter(future_layers, layer(i+layer{i}.prev));
                    case 'beamforming'
                        %             if strcmpi(future_layer.name, 'power')  % we implement the gradient of beamforming and power spectrum together for simplicity
                        %                 layer{i}.grad = B_beamforming_power(layer(i+layer{i}.next+future_layer.next), layer{i}, layer(i+layer{i}.prev));
                        %             else
                        layer{i}.grad = B_beamforming(future_layers, prev_layers, layer{i});
                        %             end
                    case 'beamforming_freeweight'
                        [layer{i}.grad, layer{i}.grad_W] = B_beamforming_freeWeight(future_layers, prev_layers{1}, layer{i});
                    case 'beamforming_gaintdoa'
                        [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_beamforming_gainTDOA(future_layers, prev_layers{1}, layer{i});
                    case 'tdoa2weight'
                        %             beamform_layer = layer{i+layer{i}.next};
                        %             [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
                        %             power_layer = layer{i+beamform_layer.next+layer{i}.next};
                        %             after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
                        %             layer{i}.grad = B_tdoa2weight_beamforming_power(X, beamform_layer, after_power_layer, layer{i});
                        layer{i}.grad = B_tdoa2weight(future_layers, layer{i});
                    case 'real_imag2bfweight'
                        beamform_layer = layer{i+layer{i}.next};
                        [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
                        power_layer = layer{i+beamform_layer.next+layer{i}.next};
                        after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
                        layer{i}.grad = B_real_imag2BFweight_beamforming_power(X, beamform_layer, after_power_layer, layer{i}, layer{i-1}.a);
                        % layer{i}.grad = B_real_imag2BFweight(layer{i+layer{i}.next}.grad, size(layer{i+layer{i}.prev}.a,2));
                    case 'realimag2complex'
                        layer{i}.grad = B_realImag2complex(future_layers, layer{i});
                    case 'spatialcovmask'
                        layer{i}.grad = B_SpatialCovMask(future_layers, layer(i+layer{i}.prev), layer{i});
                    case 'spatialcovsplitmask'
                        layer{i}.grad = B_SpatialCovSplitMask(future_layers, layer(i+layer{i}.prev), layer{i});
                    case 'mvdr_spatialcov'
                        beamform_layer = layer{i+layer{i}.next};
                        [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
                        power_layer = layer{i+beamform_layer.next+layer{i}.next};
                        after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
                        layer{i}.grad = B_MVDR_spatialCov(X, layer{i}, beamform_layer, after_power_layer);
                        
                        % other non-updatable layers
                    case 'relu'
                        layer{i}.grad = B_relu(future_layers, layer{i});
                    case 'maxout'
                    case 'mu_law'
                        layer{i}.grad = B_mu_law(future_layers, layer{i});
                    case 'tanh'
                        layer{i}.grad = B_tanh(future_layers, layer{i});
                    case {'sigmoid'}
                        layer{i}.grad = B_sigmoid(future_layers, layer{i});
                    case {'exp'}
                        layer{i}.grad = B_exp(future_layers, layer{i});
                    case 'softmax'
                        future_layer = layer{i+layer{i}.next};      % we only allow one future layer connected to softmax
                        if strcmpi(future_layer.name, 'cross_entropy')  % it is necessary to compute the gradient of
                            layer{i}.grad = B_softmax_cross_entropy(layer(i+layer{i}.next+future_layer.prev), future_layer);  % softmax and cross-entropy together to avoid numerical instability problem
                            layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i+layer{i}.next);
                        else
                            layer{i}.grad = B_softmax(layer(i+layer{i}.next), layer{i});
                        end
                    case 'multi_softmax'
                        layer{i}.grad = B_multi_softmax_multi_cross_entropy(layer(i+layer{i}.next+future_layers{1}.prev), future_layers{1});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i+layer{i}.next);
                    case 'cosine'
                        layer{i}.grad = B_cosine(prev_layers, future_layers);
                        
                    case 'cov'
                        layer{i}.grad = B_cov(layer{i+layer{i}.prev}.a, future_layers);
                    case 'logdet'
                        layer{i}.grad = B_logdet(layer{i+layer{i}.prev}.a);
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                    case 'll_gmm'
                        layer{i}.grad = B_ll_gmm(layer{i+layer{i}.prev}.a, layer{i});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                        
                    case 'll_gaussian'
                        layer{i}.grad = B_ll_gaussian(prev_layers, layer{i});
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
                        
                    case 'inner_product_normalized'
                        layer{i}.grad = B_inner_product_normalized(prev_layers, future_layers);
                    case 'concatenate'
                        layer{i}.grad = B_concatenate(prev_layers, layer{i}, future_layers);
                    case 'extractdims'
                        layer{i}.grad = B_ExtractDims(prev_layers, layer{i}, future_layers);
                        
                    otherwise
                        fprintf('Error: unknown output node type %s!\n', layer{i}.name);
                end
                if isfield(layer{i},'grad_W') && layer{i}.update && para.NET.L2weight>0
                    if issparse(layer{i}.grad_W)
                        layer{i}.grad_W = AddSpMatMat_sparseonly(1, layer{i}.grad_W, para.NET.L2weight, layer{i}.W);
                    else
                        if isfield(layer{i}, 'W0')
                            layer{i}.grad_W = layer{i}.grad_W + para.NET.L2weight * (layer{i}.W - layer{i}.W0);
                        else
                            layer{i}.grad_W = layer{i}.grad_W + para.NET.L2weight * layer{i}.W;
                        end
                    end
                end
                if para.DEBUG
                    if isfield(layer{i},'grad');
                        if iscell(layer{i}.grad)
                            for grad_i = 1:length(layer{i}.grad)
                                hasnan = hasnan + sum(sum(isnan(layer{i}.grad{grad_i})));
                            end
                        else
                            hasnan = hasnan + sum(sum(isnan(layer{i}.grad)));
                        end
                    end
                    if isfield(layer{i},'grad_b'); hasnan = hasnan + sum(sum(isnan(layer{i}.grad_b))); end
                    if isfield(layer{i},'grad_W'); hasnan = hasnan + sum(sum(isnan(layer{i}.grad))); end
                end
            end
            if para.DEBUG
                hasnan = gather(hasnan);
                if hasnan>0
                    fprintf('NAN detected in weights\n');
                elseif hasinf>0
                    fprintf('Inf detected in weights\n');
                end
            end
            if isfield(para, 'DEBUG_PLOT') && para.DEBUG_PLOT
                TunableLayer = [];
                for i=1:length(layer)
                    if isfield(layer{i},'grad_W') && layer{i}.update
                        TunableLayer(end+1) = i;
                    end
                end
                nTunableLayer = length(TunableLayer);
                for i=1:nTunableLayer
                    subplot(nTunableLayer,3,(i-1)*3+1); imagesc(layer{TunableLayer(i)}.W); colorbar
                    title(sprintf('Layer %d - %s: W', TunableLayer(i), layer{TunableLayer(i)}.name));
                    
                    subplot(nTunableLayer,3,(i-1)*3+2); imagesc(layer{TunableLayer(i)}.a); colorbar
                    title(sprintf('Layer %d - %s: a', TunableLayer(i), layer{TunableLayer(i)}.name));
                    
                    subplot(nTunableLayer,3,(i-1)*3+3); imagesc(layer{TunableLayer(i)}.grad_W); colorbar
                    title(sprintf('Layer %d - %s: grad_W', TunableLayer(i), layer{TunableLayer(i)}.name));
                end
                pause(.01)
            end
            
            % if para.DEBUG
            %     figure(2);
            %     imagesc(layer{4}.grad)
            %     pause
            % end
            
            if 0    % for debugging purpose, check whether precision is violated
                for i=1:length(layer)
                    if isfield(layer{i}, 'a')
                        fprintf('layer %d %s activation has precision "%s"\n', i, layer{i}.name, class(gather(layer{i}.a(1))));
                    end
                    if isfield(layer{i}, 'grad')
                        fprintf('layer %d %s gradient has precision "%s"\n', i, layer{i}.name,  class(gather(layer{i}.grad(1))));
                    end
                    if isfield(layer{i}, 'grad_W') && ~isempty(layer{i}.grad_W)
                        fprintf('layer %d %s gradient has precision "%s"\n', i, layer{i}.name,  class(gather(layer{i}.grad_W(1))));
                    end
                    if isfield(layer{i}, 'grad_b') && ~isempty(layer{i}.grad_b)
                        fprintf('layer %d %s gradient has precision "%s"\n', i, layer{i}.name,  class(gather(layer{i}.grad_b(1))));
                    end
                end
            end
            
        end
        
        
        
        function verifyGradient(obj, batch_data, para)
            for si=1:length(data)
                data{si} = double(data{si});
            end
            if nargin<4
                randomParaLoc = 1;  % randomly choose parameter for test
            end
            
            [cost_func,layer] = DNN_Cost10(layer, data, para, 1);
            
            EPSILON = 10^(-4);
            for i=length(layer):-1:1
                if isfield(layer{i},'W') && layer{i}.update
                    [m,n] = size(layer{i}.W);
                    layer_grad{i}.gradW_theo = zeros(m,n);
                    layer_grad{i}.gradW_num = zeros(m,n);
                    if randomParaLoc
                        nPara = 5;
                    else
                        if isfield(layer{i}, 'mask')
                            [idx_m, idx_n] = find(layer{i}.mask==1);
                            nPara = length(idx_m);
                        else
                            nPara = m*n;
                        end
                    end
                    for j=1:nPara
                        if randomParaLoc
                            if issparse(layer{i}.grad_W)
                                [nonzero_idx1, nonzero_idx2] = find(layer{i}.grad_W);
                                rand_idx = randperm(length(nonzero_idx1));
                                idx1 = nonzero_idx1(rand_idx(1));
                                idx2 = nonzero_idx2(rand_idx(1));
                            else
                                if isfield(layer{i}, 'mask')
                                    [idx1, idx2] = find(layer{i}.mask==1);
                                    random_idx = randperm(length(idx1));
                                    idx1 = idx1(random_idx(1));
                                    idx2 = idx2(random_idx(1));
                                else
                                    idx1 = randperm(m); idx1 = idx1(1);
                                    idx2 = randperm(n); idx2 = idx2(1);
                                end
                            end
                        else
                            if isfield(layer{i}, 'mask')
                                idx1 = idx_m(j);
                                idx2 = idx_n(j);
                            else
                                idx2 = ceil(j/m);
                                idx1 = j-(idx2-1)*m;
                            end
                        end
                        
                        init_val = layer{i}.W(idx1,idx2);
                        
                        layer{i}.W(idx1,idx2) = layer{i}.W(idx1,idx2) + EPSILON;
                        [cost_func2] = DNN_Cost10(layer, data, para, 2);
                        
                        layer{i}.W(idx1,idx2) = layer{i}.W(idx1,idx2) - 2*EPSILON;
                        [cost_func1] = DNN_Cost10(layer, data, para, 2);
                        
                        num_grad = gather((cost_func2.cost-cost_func1.cost))/2/EPSILON;
                        
                        layer{i}.W(idx1,idx2) = init_val;
                        
                        if isfield(layer{i}, 'grad_W')==0
                            fprintf('Lyaer %d, %s has no grad_W\n', i, layer{i}.name);
                        end
                        
                        theo_grad = gather(full(layer{i}.grad_W(idx1,idx2)));
                        fprintf('Layer %d, W(%d,%d),[num_grad,theo_grad] = [%f, %f], diff=[%2.10f, %E]\n', ...
                            i, idx1,idx2,num_grad,theo_grad,num_grad-theo_grad, (num_grad-theo_grad)/mean(abs([num_grad theo_grad])));
                        
                        layer_grad{i}.gradW_num(idx1,idx2) = num_grad;
                        layer_grad{i}.gradW_theo(idx1,idx2) = theo_grad;
                        
                    end
                end
                if isfield(layer{i},'b') && layer{i}.update
                    [m] = length(layer{i}.b);
                    for j=1:3
                        idx1 = randperm(m); idx1 = idx1(1);
                        
                        init_val = layer{i}.b(idx1);
                        
                        layer{i}.b(idx1) = layer{i}.b(idx1) + EPSILON;
                        [cost_func2] = DNN_Cost10(layer, data, para, 2);
                        
                        layer{i}.b(idx1) = layer{i}.b(idx1) - 2*EPSILON;
                        [cost_func1] = DNN_Cost10(layer, data, para, 2);
                        
                        num_grad = gather((cost_func2.cost-cost_func1.cost))/2/EPSILON;
                        
                        layer{i}.b(idx1) = init_val;
                        
                        fprintf('Layer %d, b(%d),[num_grad,theo_grad] = [%f, %f], diff=%f\n', ...
                            i, idx1,num_grad,layer{i}.grad_b(idx1),num_grad-layer{i}.grad_b(idx1));
                    end
                end
            end
        end
        
        
        
        function obj = update(obj, para, update, itr, learning_rate)
            WeightUpdateOrder = para.NET.WeightUpdateOrder;
            
            % Use momentum
            momentum_i = min(itr, length(para.NET.momentum));
            curr_momentum = para.NET.momentum(momentum_i);
            total_weight_norm = 0;
            
            if isnan(layer{end}.a)
                % Sometimes, we get nan cost function. Then we should ignore current
                % mismatch.
                % Cases that causes nan cost function: 1) in single precision, if the
                % input of softmax is too big, it will cause nan as the exp(input) is
                % inf.
                fprintf('Warning: nan cost detected, current minibatch excluded for training!\n');
                return;
            end
            
            for i=1:length(WeightUpdateOrder)
                Lidx = WeightUpdateOrder{i};
                [~, isTranspose] = VerifyTiedLayers(layer(Lidx));
                
                % collect gradients
                grad_W = layer{Lidx(1)}.grad_W;
                for k=2:length(Lidx)
                    if isTranspose(k)
                        grad_W = grad_W + layer{Lidx(k)}.grad_W';
                    else
                        grad_W = grad_W + layer{Lidx(k)}.grad_W;
                    end
                end
                
                if para.NET.gradientClipThreshold > 0
                    grad_W = max(-para.NET.gradientClipThreshold, grad_W);
                    grad_W = min(para.NET.gradientClipThreshold, grad_W);
                end
                
                %     if para.NET.rmsprop_decay>0
                %         layer{k}.gradW_avg_square = layer{k}.gradW_avg_square * para.rmsprop_decay + ...
                %             layer{k}.grad_W.^2 * (1-para.rmsprop_decay);
                %         element_learning_rate = 1./(sqrt(layer{k}.gradW_avg_square)+para.rmsprop_damping);
                %         element_learning_rate = element_learning_rate / ...
                %             sum(sum(element_learning_rate))*numel(element_learning_rate);
                %         update{k}.W = update{k}.W * curr_momentum + ...
                %             grad_W.*element_learning_rate * learning_rate;
                %     else
                if issparse(grad_W)==0      % apply momentum only when gradient is not sparse
                    update{i}.W = update{i}.W * curr_momentum + grad_W * learning_rate;
                else
                    update{i}.W = grad_W * learning_rate;
                end
                
                if para.DEBUG;
                    weight_norm_old = mean(mean(layer{Lidx(1)}.W.^2)) * length(Lidx);
                end
                
                %     if strcmpi(layer{Lidx(1)}.name, 'LSTM')            % For LSTM, we don't update W_cc and keep it 0
                %         nCell = layer{Lidx(1)}.dim(1);
                %         update{i}.W(:, 1:nCell ) = 0;
                %     end
                
                if issparse(update{i}.W)
                    layer{Lidx(1)}.W = AddSpMatMat(-1,update{i}.W, 1, layer{Lidx(1)}.W, 0);
                else
                    layer{Lidx(1)}.W = layer{Lidx(1)}.W - update{i}.W;
                end
                
                if para.NET.weight_clip
                    % sometimes the weight will explode, so we need to add a limit to the value of the weights, e.g. +-10
                    layer{Lidx(1)}.W = max(-para.NET.weight_clip,layer{Lidx(1)}.W);
                    layer{Lidx(1)}.W = min(para.NET.weight_clip,layer{Lidx(1)}.W);
                end
                
                for k=2:length(Lidx)   % copy weights to other tied layers
                    if isTranspose(k)
                        layer{Lidx(k)}.W = layer{Lidx(1)}.W';
                    else
                        layer{Lidx(k)}.W = layer{Lidx(1)}.W;
                    end
                end
                
                if para.DEBUG
                    weight_norm = mean(mean(layer{Lidx(1)}.W.^2)) * length(Lidx);
                    total_weight_norm = total_weight_norm + weight_norm;
                    if weight_norm/weight_norm_old > 1.5
                        fprintf('Warning: layer %d weight norm increases too fast: old norm: %f, new norm %f\n',k,weight_norm_old, weight_norm);
                    end
                end
                
                has_bias = isfield(layer{Lidx(1)}, 'grad_b');
                
                if has_bias
                    grad_b = layer{Lidx(1)}.grad_b;
                    for k=2:length(Lidx)
                        if ~isTranspose(k)      % if the layer is a transpose of first layer, its grad_b is not used and its b won't be trained
                            grad_b = grad_b + layer{Lidx(k)}.grad_b;
                        end
                    end
                    if para.NET.rmsprop_decay>0
                        %             layer{k}.gradb_avg_square = layer{k}.gradb_avg_square * para.rmsprop_decay + ...
                        %                 layer{k}.grad_b.^2 * (1-para.rmsprop_decay);
                        %             element_learning_rate = 1./(sqrt(layer{k}.gradb_avg_square)+para.rmsprop_damping);
                        %             element_learning_rate = element_learning_rate / ...
                        %                 sum(sum(element_learning_rate))*numel(element_learning_rate);
                        %             update{k}.b = update{k}.b * curr_momentum + ...
                        %                 layer{k}.grad_b.*element_learning_rate * learning_rate;
                    else
                        if curr_momentum>0
                            update{i}.b = update{i}.b * curr_momentum + grad_b * learning_rate;
                        else
                            update{i}.b = grad_b * learning_rate;
                        end
                    end
                    layer{Lidx(1)}.b = layer{Lidx(1)}.b - update{i}.b;
                    for k=2:length(Lidx)   % copy biases to other tied layers
                        if ~isTranspose(k)
                            layer{Lidx(k)}.b = layer{Lidx(1)}.b;
                        end
                    end
                end
            end
        end
        
        
        
        function obj= cleanUp(obj)
            fields = {'a', 'grad', 'grad_W', 'grad_b', 'acc', 'weights', 'grad_W_raw', 'grad2', 'idx', 'X2', 'ft', 'it', 'ot', 'Ct_raw', 'Ct', 'Ct0', 'ht0', 'post', 'validFrameMask'};
            for i=1:length(layer)
                for j=1:length(fields)
                    if isfield(layer{i}, fields{j})
                        layer{i} = rmfield(layer{i}, fields{j});
                    end
                end
            end
        end
        
        
        
        % automatically derive the list of layers that the output of the current layer goes.
        function layer = FinishLayer(layer)
            
            for i=1:length(layer); layer{i}.next = []; end
            for i=length(layer):-1:1
                if isfield(layer{i}, 'prev')
                    for j=1:length(layer{i}.prev)
                        layer{i+layer{i}.prev(j)}.next(end+1) = -layer{i}.prev(j);
                    end
                end
            end
        end
        
        
        
        function layer_idx = ReturnLayerIdxByName(layer, layer_name)
            
            layer_idx = [];
            for i=1:length(layer)
                if strcmpi(layer{i}.name, layer_name)
                    layer_idx(end+1) = i;
                end
            end
        end
        

        
        function obj = computeGlobalCMVN(obj, Data_tr, 100, para1, layer1)
        end
        function obj = VerifyPreprocessingTree(obj, layer(1:splice_layer_idx+1), Data_tr, para1, 100)
        end
    
    
        
    end
    methods (Access = protected)
        function obj = genWeightUpdateOrder(obj, para.NET.WeightTyingSet)
            % define the set of parameters to be updated
            weight_update_order = WeightTyingSet;
            already_in_set = cell2mat(weight_update_order);
            
            for i=length(layer):-1:1
                if IsUpdatableNode(layer{i}.name)==0; continue; end
                if layer{i}.update == 0; continue; end
                if ismember(i, already_in_set); continue; end
                weight_update_order{end+1} = i;
                already_in_set(end+1) = i;
            end
        end
        
        
        
        function obj = setDNNParameterPrecision(obj, para.singlePrecision, para.useGPU)
            WeightNames = WeightNameList('all');
            for i=1:length(layer)
                for j=1:length(WeightNames)
                    if isfield(layer{i}, WeightNames{j})
                        if singlePrecision
                            layer{i}.(WeightNames{j}) = single(layer{i}.(WeightNames{j}));
                        else
                            layer{i}.(WeightNames{j}) = double(layer{i}.(WeightNames{j}));
                        end
                        if useGPU
                            layer{i}.(WeightNames{j}) = gpuArray(layer{i}.(WeightNames{j}));
                        end
                    end
                end
            end
        end
        
        
        function obj = DetermineGradientPass(layer)
            % find the parent of every layer
            parent = DetermineLayerParent(layer);
            
            skipBP = zeros(length(layer),1);
            for i=1:length(layer)
                if isfield(layer{i}, 'skipBP')
                    skipBP(i) = layer{i}.skipBP;
                end
            end
            
            update = zeros(length(layer),1);
            for i=1:length(layer)
                if isfield(layer{i}, 'update')
                    update(i) = layer{i}.update;
                end
            end
            
            for i=1:length(layer)
                if IsUpdatableNode(layer{i}.name)==0; continue; end
                if skipBP(i); continue; end
                
                layer{i}.passGradBack = sum(update(parent{i}))>0;
            end
        end
        
    end
end

