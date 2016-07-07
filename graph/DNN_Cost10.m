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
    switch lower(layer{i}.name)
        case 'ignore'
            % just pass
    	case 'input'
    		layer{i}.a = data{layer{i}.inputIdx};
        case 'idx2vec'
            layer{i}.a = F_idx2vec(layer{i+layer{i}.prev}.a, layer{i}, para.singlePrecision);
    	case 'affine'
    		prev_layer = layer{i+layer{i}.prev};
        	if strcmpi(prev_layer.name, 'input') && para.IO.sparse(prev_layer.inputIdx)
        		layer{i}.a = F_sparse_affine_transform(prev_layer.a, layer{i}.W, layer{i}.b, para.singlePrecision);
            else
                if issparse(prev_layer.a)
                    prev_layer.a = full(prev_layer.a);
                end
        		layer{i}.a = F_affine_transform(prev_layer.a, layer{i}.W, layer{i}.b);
            end
        case 'word2vec'
            layer{i}.a = F_word2vec(layer{i+layer{i}.prev}.a, layer{i}.W, para.singlePrecision);
        case 'concatenate'
            layer{i}.a = F_concatenate(layer(i+layer{i}.prev));
        case 'weighting'
            layer{i}.a = F_weighting(layer{i+layer{i}.prev}.a, layer{i}.W, layer{i}.b);
        case 'cmn'
            layer{i}.a = CMN(layer{i+layer{i}.prev}.a')';
        case 'linear'
            layer{i}.a = layer{i+layer{i}.prev}.a;
        case {'sigmoid'}
            layer{i}.a = F_sigmoid(layer{i+layer{i}.prev}.a);
        case {'tanh'}
            layer{i}.a = F_tanh(layer{i+layer{i}.prev}.a);
        case 'softmax'
            layer{i}.a = F_softmax(layer{i+layer{i}.prev}.a);
        case 'multi_softmax'
            layer{i}.a = F_multi_softmax(layer{i+layer{i}.prev}.a, layer{i}.TaskVocabSizes);
        case 'logistic'
            [layer{i}.a, layer{i}.acc] = F_logistic(layer(i+layer{i}.prev), layer{i});
        case 'cosine'
            layer{i}.a = F_cosine(layer(i+layer{i}.prev));
        case 'inner_product'
            layer{i}.a = F_inner_product(layer(i+layer{i}.prev));
        case 'inner_product_normalized'
            layer{i}.a = F_inner_product_normalized(layer(i+layer{i}.prev));
        case 'relu'
        	layer{i}.a = max(0,layer{i+layer{i}.prev}.a);
        case 'maxout'
            
        case 'mean'
            layer{i}.a = F_mean(layer{i+layer{i}.prev}.a);
            
        case 'max'
            layer{i}.a = F_max(layer{i+layer{i}.prev}.a);
            
        case 'tconv'
            [layer{i}.a, layer{i}.X2] = F_tconv(layer{i+layer{i}.prev}.a, layer{i});
        case 'tmaxpool'
            [layer{i}.a, layer{i}.idx] = F_tmaxpool(layer{i+layer{i}.prev}.a, layer{i});
            
        case 'weighted_average'
            [layer{i}.a,layer{i}.weights] = F_weighted_average(layer(i+layer{i}.prev));
        	
        case 'multisoftmax'
            layer{i}.a = F_multisoftmax(layer{i+layer{i}.prev}.a, para.classID);
    	case 'delta'
    		layer{i}.a = F_dynamic_feat(layer{i+layer{i}.prev}.a);
    	case 'log'
    		layer{i}.a = log(layer{i}.const+layer{i+layer{i}.prev}.a);
    	case 'power'
    		layer{i}.a = F_power_spectrum(layer{i+layer{i}.prev}.a);
        case 'splice'
            layer{i}.a = F_splice(layer{i+layer{i}.prev}.a, layer{i}.context);
        case 'mel'
            layer{i}.a = F_affine_transform(layer{i+layer{i}.prev}.a, layer{i}.W, layer{i}.b);
    	case 'power_split'
    		layer{i}.a = F_power_spectrum_split(layer{i+layer{i}.prev}.a);
    	case 'beamforming'
    		layer{i}.a = F_beamforming(layer(i+layer{i}.prev));
        case 'filter'
            layer{i}.a = F_filter(layer(i+layer{i}.prev));
        case 'comp_gcc'
            layer{i}.a = F_comp_gcc(layer{i+layer{i}.prev}.a, layer{i});
        case 'stft'
            layer{i}.a = F_stft(layer{i+layer{i}.prev}.a, layer{i});
            
        case 'cov'
            layer{i}.a = F_cov(layer{i+layer{i}.prev}.a);
        case 'logdet'
            layer{i}.a = F_logdet(layer{i+layer{i}.prev}.a);
        case 'll_gmm'
            layer{i} = F_ll_gmm(layer{i+layer{i}.prev}.a, layer{i});
            
    	case 'tdoa2weight'
    		layer{i}.a = F_tdoa2weight(layer{i+layer{i}.prev}.a, layer{i}.freqBin);
    	case 'real_imag2bfweight'
            if isfield(layer{i}, 'online')==0; layer{i}.online = 0; end
    		layer{i}.a = F_real_imag2BFweight(layer{i+layer{i}.prev}.a, layer{i}.freqBin, layer{i}.online);
    	case 'mse'
    		layer{i}.a = F_mean_square_error(layer(i+layer{i}.prev), layer{i}.useMahaDist, layer{i});
    	case 'cross_entropy';
    		[layer{i}.a, layer{i}.acc] = F_cross_entropy(layer(i+layer{i}.prev), layer{i});
    	case 'multi_cross_entropy';
    		[layer{i}.a, layer{i}.acc] = F_multi_cross_entropy(layer(i+layer{i}.prev), layer{i});
        case 'lstm'
            [layer{i}] = F_LSTM(layer{i+layer{i}.prev}.a, layer{i});
        otherwise
            fprintf('Error: unknown output node type %s!\n', layer{i}.name);
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
             [layer{i}.grad, layer{i}.grad_W, layer{i}.grad_b] = B_LSTM(prev_layers{1}.a, layer{i}, future_layers);

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
    		layer{i}.grad = B_power_spectrum(future_layers);
    	case 'power_split'
    		layer{i}.grad = B_power_spectrum_split(future_layers, layer{i+layer{i}.prev}.a);
    	case 'filter'
    		layer{i}.grad = B_filter(future_layers, layer(i+layer{i}.prev));
    	case 'beamforming'
            % do not implement this now
%             future_layer = layer{i+layer{i}.next};
%             if strcmpi(future_layer.name, 'power')  % we implement the gradient of beamforming and power spectrum together for simplicity
%                 layer{i}.grad = B_beamforming_power(layer(i+layer{i}.next+future_layer.next), layer{i}, layer(i+layer{i}.prev));
%             else
%                 layer{i}.grad = B_beamforming(layer(i+layer{i}.next));
%             end
    	case 'tdoa2weight'
            beamform_layer = layer{i+layer{i}.next};
            [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
            power_layer = layer{i+beamform_layer.next+layer{i}.next};
            after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
            layer{i}.grad = B_tdoa2weight_beamforming_power(X, beamform_layer, after_power_layer, layer{i});
%     		layer{i}.grad = B_tdoa2weight(layer{i+layer{i}.next}.grad, layer{i}, layer{i+layer{i}.prev}.a);
        case 'real_imag2bfweight'
            beamform_layer = layer{i+layer{i}.next};
            [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
            power_layer = layer{i+beamform_layer.next+layer{i}.next};
            after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
            layer{i}.grad = B_real_imag2BFweight_beamforming_power(X, beamform_layer, after_power_layer, layer{i}, size(layer{i-1}.a,2));
%             layer{i}.grad = B_real_imag2BFweight(layer{i+layer{i}.next}.grad, size(layer{i+layer{i}.prev}.a,2));

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

end
