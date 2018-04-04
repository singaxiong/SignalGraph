% Class for nodes with parameters
classdef GraphNodeUpdatable < GraphNode
    properties
        % weights, gradients, and activations
        W = [];     % weight matrix if any
        b = [];     % bias vector if any
        gradW = []; % gradient of weight if any
        gradB = []; % gradient of bias if any
        
        W0 = [];    % sometimes, we have an initial W, e.g. to compute L2 cost as ||W-W0||^2
        WMask = [];     % sometimes, we have a mask of weight, which specifies whether a particular weight is used or not
        
        % configurations
        updateWeight = 1; % whether to update the weights. If set to 0, will skip the computation of gradW
        updateBias = 1; % whether to update the bias. If set to 0, will skip the computation of gradB
        L2weight = 0;   % weight of L2 norm
    end
    
    methods
        function obj = GraphNodeUpdatable(name, dimOut)
            obj = obj@GraphNode(name, dimOut);
        end
%         function obj = initialize(obj)
%         end
        
        function obj = setPrecision(obj, precision, useGPU)
            WeightNames = {'W', 'b'};
            for i=1:length(WeightNames)
                if isprop(obj, WeightNames{i})
                    if strcmpi(precision, 'single')
                        obj.(WeightNames{i}) = single(obj.(WeightNames{i}));
                    else
                        obj.(WeightNames{i}) = double(obj.(WeightNames{i}));
                    end
                    if useGPU
                        obj.(WeightNames{i}) = gpuArray(obj.(WeightNames{i}));
                    else
                        obj.(WeightNames{i}) = gather(obj.(WeightNames{i}));
                    end
                end
            end
        end
        
        function obj = setUpdate(obj, updateWeight, updateBias)
            obj.updateWeight = updateWeight;
            obj.updateBias = updateBias;
        end
        
        function obj = forward(obj, prev_layers)
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj, future_layers, prev_layers)
            % compute the gradient due to L2 weight
            if obj.updateWeight
                if issparse(obj.gradW)
                    obj.gradW = AddSpMatMat_sparseonly(1, obj.gradW, obj.L2weight, obj.W);
                else
                    if ~isempty(obj.W0)
                        obj.gradW = obj.gradW + obj.L2weight * (obj.W - obj.W0);
                    else
                        obj.gradW = obj.gradW + obj.L2weight * obj.W;
                    end
                end
            end
            obj = backward@GraphNode(obj, future_layers, prev_layers);
        end
        
        function cost = L2cost(obj)
            if obj.L2weight>0
                weight = obj.L2weight;
                if IsInGPU(obj.W)
                    weight = gpuArray(obj.L2weight);
                end
                if ~isempty(obj.W0)
                    diff = obj.W - obj.W0;   % if we are given a initial weight matrix W0, we measure the difference between W and W0.
                else
                    diff = obj.W;   % otherwise, we measure the difference between W and a matrix of zeros.
                end
                if ~isempty(obj.WMask)        % the mask defines what values can be tuned and what cannot be tuned.
                    diff = diff .* obj.WMask;
                end
                cost = 0.5* weight * sum(sum(diff.*diff));
            else
                cost = 0;
            end
        end
        
        function obj = cleanUp(obj)
            obj = cleanUp@GraphNode(obj);
            obj.gradW = [];
            obj.gradB = [];
        end
    end
    
    methods (Access = protected)
    end
    
end