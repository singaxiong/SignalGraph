classdef SignalGraph
    properties
        % configuration
        precision = 'single';   % choose [single|double]
        useGPU = 1;             % whether to use GPU
        
        costLayerIdx = 1;
        costWeight = 1;
        
        mvnLayerIdx = [];   % stores the layer index of global mean and variance normalization
        
        % data
        layer={};
        WeightTyingSet = {};
        WeightUpdateOrder = [];
    end
    methods (Static)
        function DisplayConnection(layer)
            fprintf('Layer index\tLayer name\t\t\t\tPrev[rel.|abs.]\t\t\tNext[rel.|abs.]\t\t\tDims\n');
            for i=1:length(layer)
                if strcmpi(layer{i}.name, 'input') || strcmpi(layer{i}.name, 'Weight2Activation')
                    absolutePrev = num2str(layer{i}.prev);
                else
                    absolutePrev = num2str(layer{i}.prev+i);
                end
                absoluteNext = num2str(layer{i}.next+i);
                
                nameStr = repmat(' ', 1,20); 
                nameStr(1:length(layer{i}.name)) = layer{i}.name;
                
                prevStr = sprintf('[%s | %s]', num2str(layer{i}.prev), absolutePrev);
                prevStr(20) = ' ';
                
                nextStr = sprintf('[%s | %s]', num2str(layer{i}.next), absoluteNext);
                nextStr(20) = ' ';

                fprintf('%d\t\t\t%s\t%s\t%s\t[%s]\n', i, nameStr, prevStr, nextStr, regexprep(num2str(layer{i}.dim), '\s+', ' '));
            end
        end
        
    end
    methods
        function obj = SignalGraph(precision, useGPU)
            obj.precision = precision;
            obj.useGPU = useGPU;
        end
        
        function DisplayLayerWeight(obj, layerIdx)
            currLayer = obj.layer{layerIdx};
            imagesc(currLayer.W(:,:));
            title(sprintf('Layer %d, %s, [%s]', layerIdx, currLayer.name, num2str(size(currLayer.W))));
        end
        
        function DisplayLayerActivation(obj, layerIdx)
            currLayer = obj.layer{layerIdx};
            a = currLayer.a;
            [D(1) D(2) D(3) D(4)] = size(a);
            if D(1)==1 && D(2) == 1
                plot(a(:)); 
            else
                imagesc(a(:,:));
            end
            title(sprintf('Layer %d, %s, [%s]', layerIdx, currLayer.name, num2str(size(currLayer.a))));
        end
        
        function DisplayWeight(obj)
            fprintf('Layer index\tLayer name\t\t\t\tPrev[rel.|abs.]\t\t\tNext[rel.|abs.]\t\t\tDims\t\t\tWeight\n');
            for i=1:length(obj.layer)
                if strcmpi(obj.layer{i}.name, 'input') || strcmpi(obj.layer{i}.name, 'Weight2Activation')
                    absolutePrev = num2str(obj.layer{i}.prev);
                else
                    absolutePrev = num2str(obj.layer{i}.prev+i);
                end
                absoluteNext = num2str(obj.layer{i}.next+i);
                
                nameStr = repmat(' ', 1,20);
                nameStr(1:length(obj.layer{i}.name)) = obj.layer{i}.name;
                
                prevStr = sprintf('[%s | %s]', num2str(obj.layer{i}.prev), absolutePrev);
                prevStr(20) = ' ';
                
                nextStr = sprintf('[%s | %s]', num2str(obj.layer{i}.next), absoluteNext);
                nextStr(20) = ' ';
                
                if isprop(obj.layer{i}, 'W')
                    sizeWstr = regexprep(num2str(size(obj.layer{i}.W)), '\s+', ' ');
                else
                    sizeWstr = '';
                end
                
                fprintf('%d\t\t\t%s\t%s\t%s\t[%s]\t\t[%s]\n', i, nameStr, prevStr, nextStr, regexprep(num2str(obj.layer{i}.dim), '\s+', ' '), sizeWstr);
            end
        end
        
        function PlotGraph(obj)
            % build graph from layers
            source_nodes = {};
            target_nodes = {};
            weights = [];
            for i=1:length(obj.layer)
                for j=1:length(obj.layer{i}.next)
                    source_nodes{end+1} = [num2str(i) ' ' obj.layer{i}.name ' [' regexprep(num2str(obj.layer{i}.dim(1:2)), '\s+', ' ') ']'];
                    target_idx = i+obj.layer{i}.next(j);
                    target_nodes{end+1} = [num2str(target_idx) ' ' obj.layer{target_idx}.name ' [' regexprep(num2str(obj.layer{target_idx}.dim(1:2)), '\s+', ' ') ']'];
                    weights(end+1) = 1;
                end
            end
            G = digraph(source_nodes, target_nodes, weights);
            plot(G);
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
            
            for i=1:length(obj.WeightTyingSet)    % not tested yet
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
            
            % determine whether a node needs to pass gradient back
            obj = obj.DetermineGradientPass();
            
            % set weight update order
            obj = obj.genWeightUpdateOrder();
            % 
            
        end
        
        % initialize the global MVN layers
        function obj = initializeMVN(obj, data, para, nUttUsed)
            if exist('nUttUsed')==0 || isempty(nUttUsed)
                nUttUsed = 500;
            end
            nUtt = length(data.streams(1).data);
            if nUtt>nUttUsed
                step = ceil(nUtt/nUttUsed);
                for i=1:data.nStream
                    data.streams(i).data = data.streams(i).data(1:step:end);
                end
            end
            
            mvnLayerIdxSorted = sort(obj.mvnLayerIdx);
            for i=1:length(mvnLayerIdxSorted)     % if there are multiple mvn layers, process the lower numbered one first as this is the layers to be processed first in forward pass
                prev = obj.layer{mvnLayerIdxSorted(i)}.prev + mvnLayerIdxSorted(i);
                [output, mask] = processData(obj, data, para, prev);
                
                for j=1:size(output,2)
                    featTmp = gather(output{1,j});
                    [featTmp2, mask, variableLength] = ExtractVariableLengthTrajectory(featTmp);
                    feat{j} = cell2mat(featTmp2);
                end
                featAll = cell2mat(feat);
                
                std_dev = std(featAll,[], 2);
                idx = find(std_dev==0);
                std_dev(idx) = 1;       % if a dimension has zero variance, set its deviation to 1 to avoid division by 0. 
                W = diag(1./std_dev);
                transformed = W * featAll;
                b = -mean(transformed,2);
                
                obj.layer{mvnLayerIdxSorted(i)}.W = W;
                obj.layer{mvnLayerIdxSorted(i)}.b = b;
                obj = obj.setPrecision(mvnLayerIdxSorted(i));
                
                % verification
                std_dev2 = std(transformed,[], 2);
                plot(std_dev2); hold on;
                plot(mean(bsxfun(@plus, transformed, b),2)); hold off;
            end
        end
         
        % Given data, get the activitation of the network at specificied
        % nodes. 
        function [output, mask] = processData(obj, data, para, outLayerIdx)
            for i=max(outLayerIdx)+1 : length(obj.layer)    % skip the layers after the last output layer, which is not necessary to generate the outputs
                obj.layer{i}.skip = 1;
            end
            
            data = data.ShuffleData(para.IO.maxNumSentInBlock);
            randomOrder = 0;
            
            output = {}; mask = {};
            cnt = 0;
            for blk_i = 1:data.nBlock
                minibatch = data.PrepareMinibatch(blk_i,randomOrder);
                nMinibatch = size(minibatch,2);
                
                for batch_i = 1:nMinibatch
                    PrintProgress(batch_i, nMinibatch, 100, 'SignalGraph:processData');
                    mb = data.GetOneMinibatch(minibatch, batch_i, para.useGPU);
                    obj = obj.forward(mb);
                    cnt = cnt + 1;
                    for j=1:length(outLayerIdx)
                        output{j, cnt} = obj.layer{outLayerIdx(j)}.a;
                        mask{j,cnt} = obj.layer{outLayerIdx(j)}.mask;
                    end
                end
            end
        end

        function [obj] = forward(obj, batch_data)            % Run forward propogation
            for i=1:length(obj.layer)
                if obj.layer{i}.skip; continue; end
                if obj.layer{i}.prev>0
                    prev_layers = {}; 
                else
                    prev_layers = obj.layer(i+obj.layer{i}.prev);    
                end
                switch lower(obj.layer{i}.name)
                    case 'ignore'
                        % do nothing
                    case {'input'}
                        obj.layer{i}.a = batch_data{obj.layer{i}.prev};
                    case {'weight2activation'}
                        obj.layer{i}.a = obj.layer{i}.W;
                    otherwise
                        obj.layer{i} = obj.layer{i}.forward(prev_layers);
                end
            end
        end
        
        function cost = evalCost(obj)                       % compute the cost function
            nCost = length(obj.costLayerIdx);
            cost = GraphCost(nCost, 1);
            
            for i=1:nCost
                currCostLayer = obj.layer{obj.costLayerIdx(i)};
                cost.subCost(i) = gather(obj.costWeight(i) * currCostLayer.a);
                
                if strcmpi(currCostLayer.name, 'cross_entropy') || strcmpi(currCostLayer.name, 'logistic') || strcmpi(currCostLayer.name, 'multi_cross_entropy')
                    cost.subAcc(i) = gather(obj.layer{obj.costLayerIdx(i)}.acc);     % For classification cost functions, also report the classification accuracy
                else
                    cost.subAcc(i) = -1;
                end
            end
            cost.taskCost = sum(cost.subCost);
            % compute the L2 regularization cost
            reguCost = 0;
            for i=1:length(obj.layer)
                if isprop(obj.layer{i}, 'L2weight') && obj.layer{i}.L2weight>0 && obj.layer{i}.updateWeight
                    reguCost = reguCost + obj.layer{i}.L2cost();
                end
            end
            cost.reguCost = gather(reguCost);
            cost.totalCost = cost.taskCost + cost.reguCost;
        end
            
        function obj = backward(obj)
            for i=length(obj.layer):-1:1
                if obj.layer{i}.skip; continue; end
                if strcmpi(obj.layer{i}.name, 'input'); continue; end       % input layer do not have backpropagation
                if obj.layer{i}.skipBP == 1; continue; end   % some layers do not need to compute gradients, such as comp_gcc and stft, or does not implement BP. 
                
                prev_layers = obj.layer(i+obj.layer{i}.prev);
                future_layers = obj.layer(i+obj.layer{i}.next);
                
                switch lower(obj.layer{i}.name)
                    case {'multicrossentropy', 'crossentropy'}    %compute the gradient together with softmax
                    case {'softmax', 'multisoftmax'}
                        if length(future_layers)==1 && strcmpi(future_layers{1}.name, 'CrossEntropy') || strcmpi(future_layers{1}.name, 'multicrossentropy')
                            target_layer = obj.layer{i+obj.layer{i}.next+future_layers{1}.prev(2)};
                            obj.layer{i} =  obj.layer{i}.backward(prev_layers, future_layers, target_layer.a);
                        else
                            obj.layer{i} =  obj.layer{i}.backward(prev_layers, future_layers);
                        end
                    otherwise
                        obj.layer{i} =  obj.layer{i}.backward(prev_layers, future_layers);
                end
            end
        end
        
        function obj = setGlobalL2weight(obj, L2weight)
            for i=1:length(obj.layer)
                if isprop(obj.layer{i}, 'L2weight')
                    obj.layer{i}.L2weight = L2weight;
                end
            end
        end
        
        function verifyGradient(obj, data, randomParaLoc)
            for si=1:length(data)
                data{si} = double(data{si});
            end
            if nargin<3
                randomParaLoc = 1;  % randomly choose parameter for test
            end
            obj.precision = 'double';
            obj = obj.setPrecision();
            
            obj = obj.forward(data);
            % cost = obj.evalCost();
            obj = obj.backward();
            
            % [cost_func,layer] = DNN_Cost10(layer, data, para, 1);
            
            EPSILON = 10^(-4);
            for i=length(obj.layer):-1:1
                if isprop(obj.layer{i},'W') && obj.layer{i}.updateWeight
                    [m,n] = size(obj.layer{i}.W);
                    layer_grad{i}.gradW_theo = zeros(m,n);
                    layer_grad{i}.gradW_num = zeros(m,n);
                    
                    if randomParaLoc
                        nPara = 5;
                    else
                        if isfield(obj.layer{i}, 'mask')
                            [idx_m, idx_n] = find(obj.layer{i}.mask==1);
                            nPara = length(idx_m);
                        else
                            nPara = m*n;
                        end
                    end
                    for j=1:nPara
                        if randomParaLoc    % randomly pic a few parameters for verification
                            if issparse(obj.layer{i}.gradW)
                                [nonzero_idx1, nonzero_idx2] = find(obj.layer{i}.gradW);
                                rand_idx = randperm(length(nonzero_idx1));
                                idx1 = nonzero_idx1(rand_idx(1));
                                idx2 = nonzero_idx2(rand_idx(1));
                            else
                                if isprop(obj.layer{i}, 'weightMask') && ~isempty(obj.layer{i}.weightMask)  % weight mask defines the used weight. E.g. if we want to use a diagonal transform, the weightMask is an identify matrix
                                    [idx1, idx2] = find(obj.layer{i}.mask==1);
                                    random_idx = randperm(length(idx1));
                                    idx1 = idx1(random_idx(1));
                                    idx2 = idx2(random_idx(1));
                                else
                                    idx1 = randperm(m); idx1 = idx1(1);
                                    idx2 = randperm(n); idx2 = idx2(1);
                                end
                            end
                        else
                            if isfield(obj.layer{i}, 'mask')
                                idx1 = idx_m(j);
                                idx2 = idx_n(j);
                            else
                                idx2 = ceil(j/m);
                                idx1 = j-(idx2-1)*m;
                            end
                        end
                        
                        init_val = obj.layer{i}.W(idx1,idx2);
                        
                        obj.layer{i}.W(idx1,idx2) = obj.layer{i}.W(idx1,idx2) + EPSILON;
                        obj = obj.forward(data);
                        cost2 = obj.evalCost();
                        
                        obj.layer{i}.W(idx1,idx2) = obj.layer{i}.W(idx1,idx2) - 2*EPSILON;
                        obj = obj.forward(data);
                        cost1 = obj.evalCost();
                        
                        num_grad = gather((cost2.totalCost-cost1.totalCost))/2/EPSILON;
                        
                        obj.layer{i}.W(idx1,idx2) = init_val;
                        
                        if isprop(obj.layer{i}, 'gradW')==0
                            fprintf('Layer %d, %s has no gradW\n', i, obj.layer{i}.name);
                        end
                        
                        theo_grad = gather(full(obj.layer{i}.gradW(idx1,idx2)));
                        fprintf('Layer %d, W(%d,%d),[num_grad,theo_grad] = [%f, %f], diff=[%2.10f, %E]\n', ...
                            i, idx1,idx2,num_grad,theo_grad,num_grad-theo_grad, (num_grad-theo_grad)/mean(abs([num_grad theo_grad])));
                        
                        layer_grad{i}.gradW_num(idx1,idx2) = num_grad;
                        layer_grad{i}.gradW_theo(idx1,idx2) = theo_grad;
                        
                    end
                end
                if isprop(obj.layer{i},'b') && obj.layer{i}.updateBias
                    [m] = length(obj.layer{i}.b);
                    for j=1:3
                        idx1 = randperm(m); idx1 = idx1(1);
                        
                        init_val = obj.layer{i}.b(idx1);
                        
                        obj.layer{i}.b(idx1) = obj.layer{i}.b(idx1) + EPSILON;
                        obj = obj.forward(data);
                        cost2 = obj.evalCost();
                        
                        obj.layer{i}.b(idx1) = obj.layer{i}.b(idx1) - 2*EPSILON;
                        obj = obj.forward(data);
                        cost1 = obj.evalCost();
                        
                        num_grad = gather((cost2.totalCost-cost1.totalCost))/2/EPSILON;
                        
                        obj.layer{i}.b(idx1) = init_val;
                        
                        fprintf('Layer %d, b(%d),[num_grad,theo_grad] = [%f, %f], diff=%f\n', ...
                            i, idx1,num_grad,obj.layer{i}.gradB(idx1),num_grad-obj.layer{i}.gradB(idx1));
                    end
                end
            end
        end
        
        
        function obj= cleanUp(obj)
            for i=1:length(obj.layer)
                obj.layer{i} = obj.layer{i}.cleanUp();
            end
        end
        
        
        
        % automatically derive the list of layers that the output of the current layer goes.
        function obj = finishGraph(obj)
%             for i=1:length(obj.layer); obj.layer{i}.next = []; end
%             for i=length(obj.layer):-1:1
%                 if isprop(obj.layer{i}, 'prev') && ~strcmpi(obj.layer{i}.name, 'input')
%                     prev = obj.layer{i}.prev;
%                     for j=1:length(prev)
%                         obj.layer{i+prev(j)}.next(end+1) = -prev(j);
%                     end
%                 end
%             end
        end
        
        
        
        function layer_idx = ReturnLayerIdxByName(layer, layer_name)
            layer_idx = [];
            for i=1:length(layer)
                if strcmpi(layer{i}.name, layer_name)
                    layer_idx(end+1) = i;
                end
            end
        end
        
        function obj = setPrecision(obj, layerIdx)
            if nargin<2 || isempty(layerIdx)
                layerIdx = 1:length(obj.layer);
            end
            for i=1:length(layerIdx)
                if ismethod(obj.layer{i}, 'setPrecision')
                    obj.layer{i} = obj.layer{i}.setPrecision(obj.precision, obj.useGPU);
                end
            end
        end
        
    end
    methods (Access = protected)
        function obj = genWeightUpdateOrder(obj)
            % define the set of parameters to be updated
            obj.WeightUpdateOrder = obj.WeightTyingSet;
            already_in_set = cell2mat(obj.WeightUpdateOrder);
            
            for i=length(obj.layer):-1:1
                if ~isprop(obj.layer{i}, 'updateWeight'); continue; end
                if obj.layer{i}.updateWeight + obj.layer{i}.updateBias == 0 || obj.layer{i}.skipBP; continue; end
                if ismember(i, already_in_set); continue; end
                obj.WeightUpdateOrder{end+1} = i;
                already_in_set(end+1) = i;
            end
        end
        
        % determine whether to compute gradient or skip the BP altogether
        % for each node. This will reduce computation in backward pass. 
        % skipBP means to skip BP. 
        % skipGrad means to only skip the passing back of gradient, but may
        % still update the weights. E.g. for the first updatable node inthe
        % graph. 
        function obj = DetermineGradientPass(obj)
            % 1. find the parent of every layer
            parent = obj.DetermineLayerParent();
            
            % 2. find out which nodes require parameter update
            update = zeros(length(obj.layer),1);
            for i=1:length(obj.layer)
                if isprop(obj.layer{i}, 'updateWeight')
                    update(i) = obj.layer{i}.updateWeight + obj.layer{i}.updateBias;
                end
            end
            
            % 3. determine whether to skip a node in BP
            for i=1:length(obj.layer)
                hasUpdatingParent = sum(update(parent{i}))>0;
                if hasUpdatingParent                % if has parent that needs update, do not skip BP or gradient passing
                    obj.layer{i}.skipBP = 0;
                    obj.layer{i}.skipGrad = 0;
                else
                    obj.layer{i}.skipGrad = 1;      % if no parent node needs to be updated, skip gradient passing
                    obj.layer{i}.skipBP = update(i)==0;     % also skip BP if current node do not need to be updated. 
                end
            end
            
%             % if a node has skipBP=1, set all its parent nodes' skipBP to 1
%             for i=1:length(obj.layer)
%                 if skipBP(i)==1
%                     for j=1:length(parent{i})
%                         skipBP(parent{i}(j)) = 1;
%                     end
%                 end
%             end
%             
%             % re-assign skipBP to layers, and also update the updateWeight
%             % and updateBias
%             for i=1:length(obj.layer)
%                 obj.layer{i}.skipBP = skipBP(i);
%                 if skipBP(i)==1
%                     if isprop(obj.layer{i}, 'updateWeight')
%                         obj.layer{i}.updateWeight = 0;
%                     end
%                     if isprop(obj.layer{i}, 'updateBias')
%                         obj.layer{i}.updateBias = 0;
%                     end
%                 end
%             end
           
            
%             for i=1:length(obj.layer)
%                 if IsUpdatableNode(obj.layer{i}.name)==0; continue; end
%                 if skipBP(i); continue; end
%                 
%                 obj.layer{i}.skipGrad = sum(update(parent{i}))==0;      % if no paraent node need to be updated, don't compute the gradient
%             end
        end
        
        % find the parent of every layer
        function parent = DetermineLayerParent(obj)
            for i=1:length(obj.layer)
                switch lower(obj.layer{i}.name)
                    case {'weight2activation', 'input'}
                        parent{i} = [];
                    otherwise
                        immediateParent = i+obj.layer{i}.prev;
                        parent{i} = [];
                        for j=immediateParent
                            parent{i} = [parent{i} j parent{j}];
                        end
                end
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

