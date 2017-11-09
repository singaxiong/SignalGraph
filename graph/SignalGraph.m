classdef SignalGraph
    properties
        % configuration
        precision = 'single';   % choose [single|double]
        useGPU = 1;             % whether to use GPU
        
        % data
        layer={};
        WeightTyingSet = {};
        WeightUpdateOrder = [];
    end
    methods
        function obj = SignalGraph()
        end
        
        
        
        function obj = initialize(obj, useGaussInit, useNegbiasInit)
            if nargin<3; useNegbiasInit = 0; end
            if nargin<2; useGaussInit = 0; end
            
            % Initialize parameters randomly based on layer sizes.
            nNode = 0;
            for i=1:length(obj.layer)
                if strcmpi(obj.layer{i}.name, 'affine') || strcmpi(obj.layer{i}.name, 'input') || strcmpi(obj.layer{i}.name, 'word2vec')
                    nNode = nNode + obj.layer{i}.dim(1);
                end
            end
            
            r  = sqrt(6) / sqrt(nNode);   % we'll choose weights uniformly from the interval [-r, r]
            for i=1:length(obj.layer)
                if ismethod(obj.layer{i}, 'initialize')
                    obj.layer{i} = obj.layer{i}.initialize(useGaussInit, useNegbiasInit, r);
                end
            end
            
            for i=1:length(obj.WeightTyingSet)
                currTyingSet = obj.WeightTyingSet{i};
                [dimMismatch, isTranspose] = VerifyTiedLayers(obj.layer(currTyingSet));
                
                baseNode = obj.layer{currTyingSet(1)};
                for j=2:length(currTyingSet)
                    if isfield(baseNode, 'W')
                        if isTranspose(j)
                            obj.layer{currTyingSet(j)}.W = baseNode.W';
                        else
                            obj.layer{currTyingSet(j)}.W = baseNode.W;
                        end
                    end
                    if isfield(baseNode, 'b')   && ~isTranspose(j)  % if the two shared layers are transpose of each other, we don't share the bias
                        obj.layer{currTyingSet(j)}.b = baseNode.b;
                    end
                end
            end
            
            % set parameter precision
            obj = obj.setPrecision();
            % set weight update order 
            obj = obj.genWeightUpdateOrder();
            % 
            obj.layer = DetermineGradientPass(obj.layer);
            
        end
         
        
        
        
        function [cost_func_tmp, obj] = forwardBackward(obj, batch_data, para, mode)
            layer = obj.layer;
            precision = para.precision;
            
            % Run forward propogation
            for i=1:obj.nLayer
                if isfield(layer{i}, 'prev')
                    prev_layers = layer(i+layer{i}.prev);    
                else
                    prev_layers = {};
                end
                switch lower(layer{i}.name)
                    case 'ignore'
                        % do nothing
                    case 'input'
                        layer{i}.a = batch_data{layer{i}.prev};
                    otherwise
                        layer{i} = layer{i}.forward(prev_layers, precision);
                end
            end
            
            if mode ==3     % do forward pass only
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
            if para.useGPU
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
            
            % Vectorized implementation of backpropagation
            if para.DEBUG; hasnan=0; end
            for i=obj.nLayer:-1:1
                if isfield(layer{i}, 'prev');   prev_layers = layer(i+layer{i}.prev);    end
                if isfield(layer{i}, 'next');   future_layers = layer(i+layer{i}.next);    end
                if isfield(layer{i}, 'skipBP') && layer{i}.skipBP == 1; continue; end   % some layers do not need to compute gradients, such as comp_gcc and stft
                switch lower(layer{i}.name)
                    case {'input', 'idx2vec', 'enframe', 'comp_gcc', 'stft'} % do nothing
                    case {'multi_cross_entropy', 'cross_entropy'}    %compute the gradient together with softmax
                    case 'softmax'
                        future_layer = layer{i+layer{i}.next};      % we only allow one future layer connected to softmax
                        if strcmpi(future_layer.name, 'cross_entropy')  % it is necessary to compute the gradient of
                            layer{i}.grad = B_softmax_cross_entropy(layer(i+layer{i}.next+future_layer.prev), future_layer);  % softmax and cross-entropy together to avoid numerical instability problem
                            layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i+layer{i}.next);
                        else
                            layer{i}.grad = B_softmax(layer(i+layer{i}.next), layer{i});
                        end
                        
                    case 'real_imag2bfweight'
                        beamform_layer = layer{i+layer{i}.next};
                        [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
                        power_layer = layer{i+beamform_layer.next+layer{i}.next};
                        after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
                        layer{i}.grad = B_real_imag2BFweight_beamforming_power(X, beamform_layer, after_power_layer, layer{i}, layer{i-1}.a);

                    case 'mvdr_spatialcov'
                        beamform_layer = layer{i+layer{i}.next};
                        [X] = prepareBeamforming(layer(i+layer{i}.next+beamform_layer.prev));
                        power_layer = layer{i+beamform_layer.next+layer{i}.next};
                        after_power_layer = layer{i+beamform_layer.next+layer{i}.next+power_layer.next};
                        layer{i}.grad = B_MVDR_spatialCov(X, layer{i}, beamform_layer, after_power_layer);

                    otherwise
                        layer{i} =  layer{i}.backward(prev_layers, future_layers, precision);
                end
                switch lower(layer{i}.name) % for cost layers, we need to set scale to the graduent
                    case {'logistic','mse','mixture_mse','jointcost','logdet','ll_gmm','ll_gaussian','multi_softmax'}
                        layer{i}.grad = SetCostWeightOnGrad(layer{i}.grad, para.cost_func, i);
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
        
   
        
    end
    methods (Access = protected)
        function obj = genWeightUpdateOrder(obj)
            % define the set of parameters to be updated
            obj.weightUpdateOrder = obj.WeightTyingSet;
            already_in_set = cell2mat(obj.weightUpdateOrder);
            
            for i=length(obj.layer):-1:1
                if IsUpdatableNode(obj.layer{i}.name)==0; continue; end
                if obj.layer{i}.update == 0; continue; end
                if ismember(i, already_in_set); continue; end
                obj.weightUpdateOrder{end+1} = i;
                already_in_set(end+1) = i;
            end
        end
        
        
        
        function obj = setPrecision(obj)
            for i=1:length(obj.layer)
                if ismethod(obj.layer{i}, 'setPrecision')
                    obj.layer{i} = obj.layer{i}.setPrecision(obj.precision, obj.useGPU);
                end
            end
        end
        
        
        function obj = DetermineGradientPass(obj,layer)
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
        
        
        function [dimMismatch, isTranspose] = VerifyTiedLayers(tiedLayers)
            
            for j=1:length(tiedLayers)
                dim(:,j) = tiedLayers{j}.dim;
            end
            dimMismatch = [];
            isTranspose = [];
            for j=2:length(tiedLayers)
                if sum(abs(dim(:,j)-dim(:,1)))
                    if sum(abs(dim(end:-1:1,j)-dim(:,1)))
                        dimMismatch(j) = 1;
                        fprintf('Error: dimension mismatch between layers that share the same weight matrix\n');
                    else
                        isTranspose(j) = 1;
                    end
                else
                    isTranspose(j) = 0;
                end
            end
            
        end
        
    end
end

