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
function [cost_func,layer,output] = DNN_Cost10(layer, data, para, mode)
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
            layer{i}.a = F_idx2vec(prev_layers{1}.a, layer{i}, para.singlePrecision);       % do not support variable length yet
    	case 'affine'
        	if strcmpi(prev_layers{1}.name, 'input') && para.IO.sparse(prev_layers{1}.inputIdx)
        		[layer{i}.a, layer{i}.validFrameMask] = F_sparse_affine_transform(prev_layers{1}, layer{i}.W, layer{i}.b, para.singlePrecision);
            else
                if issparse(prev_layers{1}.a);  prev_layers{1}.a = full(prev_layers{1}.a); end
        		[layer{i}.a, layer{i}.validFrameMask] = F_affine_transform(prev_layers{1}, layer{i}.W, layer{i}.b);
            end
        case 'word2vec'
            layer{i}.a = F_word2vec(prev_layers{1}.a, layer{i}.W, para.singlePrecision);    % do not support variable length yet
        case 'concatenate'
            layer{i}.a = F_concatenate(prev_layers);    % do not support variable length yet
        case 'extractdims'
            layer{i}.a = F_ExtractDims(prev_layers{1}, layer{i}.dimIndex);
        case 'weighting'
            layer{i}.a = F_weighting(prev_layers{1}.a, layer{i}.W, layer{i}.b);             % do not support variable length yet
        case 'cmn'
            [layer{i}.a, layer{i}.validFrameMask] = F_cmn(prev_layers{1});
        case 'linear'
            layer{i}.a = prev_layers{1}.a;
        case {'sigmoid'}
            layer{i}.a = F_sigmoid(prev_layers{1});
        case {'tanh'}
            layer{i}.a = F_tanh(prev_layers{1});
        case 'softmax'
            layer{i}.a = F_softmax(prev_layers{1});
        case 'multi_softmax'
            layer{i}.a = F_multi_softmax(prev_layers{1}.a, layer{i}.TaskVocabSizes);
        case 'logistic'
            [layer{i}.a, layer{i}.acc] = F_logistic(prev_layers, layer{i});
        case 'cosine'
            layer{i}.a = F_cosine(prev_layers);
        case 'inner_product'
            layer{i}.a = F_inner_product(prev_layers);
        case 'inner_product_normalized'
            layer{i}.a = F_inner_product_normalized(prev_layers);
        case 'relu'
        	layer{i}.a = max(0,prev_layers{1}.a);
        case 'maxout'
            
        case 'mean'
            layer{i}.a = F_mean(prev_layers{1}.a);
            
        case 'max'
            layer{i}.a = F_max(prev_layers{1}.a);
            
        case 'tconv'
            [layer{i}.a, layer{i}.X2] = F_tconv(prev_layers{1}.a, layer{i});
        case 'tmaxpool'
            [layer{i}.a, layer{i}.idx] = F_tmaxpool(prev_layers{1}.a, layer{i});
            
        case 'weighted_average'
            [layer{i}.a,layer{i}.weights] = F_weighted_average(prev_layers);
        	
        case 'multisoftmax'
            layer{i}.a = F_multisoftmax(prev_layers{1}.a, para.classID);
    	case 'delta'
    		[layer{i}.a, layer{i}.validFrameMask] = F_dynamic_feat(prev_layers{1});
    	case 'log'
    		[layer{i}.a, layer{i}.validFrameMask] = F_log(prev_layers{1}, layer{i}.const);
    	case 'power'
    		layer{i}.a = F_power_spectrum(prev_layers{1});
        case 'splice'
            [layer{i}.a, layer{i}.validFrameMask] = F_splice(prev_layers{1}, layer{i}.context);
        case 'mel'
            [layer{i}.a, layer{i}.validFrameMask] = F_affine_transform(prev_layers{1}, layer{i}.W, layer{i}.b);
    	case 'power_split'
    		layer{i}.a = F_power_spectrum_split(prev_layers{1}.a);
    	case 'beamforming'
    		[layer{i}.a, layer{i}.validFrameMask] = F_beamforming(prev_layers);
        case 'filter'
            layer{i}.a = F_filter(prev_layers);
        case 'comp_gcc'
            [layer{i}.a, layer{i}.validFrameMask] = F_comp_gcc(prev_layers{1}, layer{i});
        case 'stft'
            [layer{i}.a, layer{i}.validFrameMask] = F_stft(prev_layers{1}, layer{i});
            
        case 'spatialcovmask'
            layer{i}.a = F_SpatialCovMask(prev_layers, layer{i});       % do not support variable length yet
        case 'mvdr_spatialcov'
            layer{i} = F_MVDR_spatialCov(prev_layers{1}, layer{i});       % do not support variable length yet
            
        case 'cov'
            layer{i}.a = F_cov(prev_layers{1}.a);       % do not support variable length yet
        case 'logdet'
            layer{i}.a = F_logdet(prev_layers{1}.a);    % do not support variable length yet
        case 'll_gmm'
            layer{i} = F_ll_gmm(prev_layers{1}.a, layer{i});    % do not support variable length yet
            
    	case 'tdoa2weight'
    		layer{i}.a = F_tdoa2weight(prev_layers{1}.a, layer{i}.freqBin);
    	case 'real_imag2bfweight'
            if isfield(layer{i}, 'online')==0; layer{i}.online = 0; end
    		[layer{i}.a, layer{i}.validFrameMask] = F_real_imag2BFweight(prev_layers{1}, layer{i}.freqBin, layer{i}.online);
    	case 'mse'
    		layer{i}.a = F_mean_square_error(prev_layers, layer{i}.useMahaDist, layer{i});
    	case 'cross_entropy';
    		[layer{i}.a, layer{i}.acc] = F_cross_entropy(prev_layers, layer{i});
    	case 'multi_cross_entropy';
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
        if para.NET.L1weight>0
            layer{i}.rho = mean(layer{i}.a,2);
        end
    end
end

if mode ==3
    for i=1:length(para.out_layer_idx)
        output{i} = layer{para.out_layer_idx(i)}.a;
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

if para.NET.L1weight>0    % If use sparsity constraint
    for i=2:nLayer-1
        if isfield(layer{i}, 'rho')
        	% note that we limit the denominator to be greater than a small number
	        tmp = para.NET.L1*log(para.NET.L1./max(1e-3,layer{i}.rho)) + (1-para.NET.L1)*log((1-para.NET.L1)./max(1e-3,1-layer{i}.rho));  
    	    cost_func.cost = cost_func.cost + para.NET.L1weight * sum(tmp);
    	end
    end
end

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
        
        % updatable layers
        case {'affine', 'mel'}
            [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_affine_transform(prev_layers, layer{i}, future_layers, i==2);
        case 'word2vec'
            [layer{i}.grad, layer{i}.grad_W] = B_word2vec(prev_layers, layer{i}, future_layers, para.singlePrecision);
        case {'weighting'}
            [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_weighting(prev_layers, layer{i}, future_layers);
        case 'lstm'
             [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_LSTM(prev_layers{1}, layer{i}, future_layers);

        % cost layers
        
        case 'mse'
    		layer{i}.grad = B_mean_square_error(prev_layers, layer{i}.useMahaDist, layer{i});
            layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
    	case {'multi_cross_entropy', 'cross_entropy'}    %compute the gradient together with softmax
    		% layer{i}.grad = B_cross_entropy(layer(i+layer{i}.prev));
        case 'logistic'
            layer{i}.grad = B_logistic(prev_layers, layer{i});
            layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
            
        % temporal layers: require sequential training
        case 'mean'
            layer{i}.grad = B_mean(prev_layers, future_layers);
            
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
            layer{i}.grad = B_splice(future_layers, layer{i}.context);
        case 'cmn'
            layer{i}.grad = B_cmn(future_layers);
            
        % signal processing layers
    	case 'log'
    		layer{i}.grad = B_log(future_layers, layer{i+layer{i}.prev}.a, layer{i}.const);
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
                layer{i}.grad = B_beamforming(future_layers, prev_layers);
%             end
    	case 'tdoa2weight'
            beamform_layer = layer{i+layer{i}.next};
            [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
            power_layer = layer{i+beamform_layer.next+layer{i}.next};
            after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
            layer{i}.grad = B_tdoa2weight_beamforming_power(X, beamform_layer, after_power_layer, layer{i});
    		% layer{i}.grad = B_tdoa2weight(layer{i+layer{i}.next}.grad, layer{i}, layer{i+layer{i}.prev}.a);
        case 'real_imag2bfweight'
            beamform_layer = layer{i+layer{i}.next};
            [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
            power_layer = layer{i+beamform_layer.next+layer{i}.next};
            after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
            layer{i}.grad = B_real_imag2BFweight_beamforming_power(X, beamform_layer, after_power_layer, layer{i}, layer{i-1}.a);
            % layer{i}.grad = B_real_imag2BFweight(layer{i+layer{i}.next}.grad, size(layer{i+layer{i}.prev}.a,2));
        case 'spatialcovmask'
            layer{i}.grad = B_SpatialCovMask(future_layers, layer(i+layer{i}.prev), layer{i});
        case 'mvdr_spatialcov'
            beamform_layer = layer{i+layer{i}.next};
            [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
            power_layer = layer{i+beamform_layer.next+layer{i}.next};
            after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
            layer{i}.grad = B_MVDR_spatialCov(X, layer{i}, beamform_layer, after_power_layer);
            
        % other non-updatable layers
        case 'relu'
        	layer{i}.grad = B_relu(future_layers, layer{i}.a);
        case 'maxout'
        case 'tanh'
            layer{i}.grad = B_tanh(future_layers, layer{i}.a);
        case {'sigmoid'}
        	layer{i}.grad = B_sigmoid(future_layers, layer{i}.a);
        case 'softmax'
            future_layer = layer{i+layer{i}.next};      % we only allow one future layer connected to softmax
            if strcmpi(future_layer.name, 'cross_entropy')  % it is necessary to compute the gradient of 
                layer{i}.grad = B_softmax_cross_entropy(layer(i+layer{i}.next+future_layer.prev), future_layer);  % softmax and cross-entropy together to avoid numerical instability problem
                layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i+layer{i}.next);
            else
                layer{i}.grad = B_softmax(future_layer.grad);
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
            
        case 'inner_product_normalized'
            layer{i}.grad = B_inner_product_normalized(prev_layers, future_layers);
        case 'concatenate'
            layer{i}.grad = B_concatenate(prev_layers, layer{i}, future_layers);

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
        if isfield(layer{i},'grad'); hasnan = hasnan + sum(sum(isnan(layer{i}.grad))); end
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
