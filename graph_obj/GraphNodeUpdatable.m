% Class for nodes with parameters
classdef GraphNodeUpdatable < GraphNode
    properties        
        % weights, gradients, and activations
        W = [];     % weight matrix if any
        b = [];     % bias vector if any
        gradW = []; % gradient of weight if any
        gradB = []; % gradient of bias if any
        
        % configurations
        updateWeight = 1; % whether to update the weights. If set to 0, will skip the computation of gradW
        updateBias = 1; % whether to update the bias. If set to 0, will skip the computation of gradB
        L2weight = 0;   % weight of L2 norm
    end
    
    methods
        function obj = GraphNodeUpdatable(name, myIdx)
            obj = obj@GraphNode(name, myIdx);
        end
        function obj = initialize(obj)
        end
        
        function obj = setPrecision(obj, precision, useGPU)
            WeightNames = {'W', 'b'};
            for i=1:length(WeightNames)
                if isfield(obj, WeightNames{i})
                    if strcmpi(precision, 'single')
                        obj.(WeightNames{i}) = single(obj.(WeightNames{i}));
                    else
                        obj.(WeightNames{i}) = double(obj.(WeightNames{i}));
                    end
                    if useGPU
                        obj.(WeightNames{i}) = gpuArray(obj.(WeightNames{i}));
                    end
                end
            end
        end
        
        function obj = forward(obj, prev_layers)
        end
        function obj = backward(obj, future_layers, prev_layers)
        end
        function obj = L2cost(obj)
        end
        function obj = update(obj)
        end
        function obj = verifyWeightAndGrad(obj)
        end
        function obj = cleanUp(obj)
            obj = cleanUp@GraphNode(obj);
            obj.W = [];
            obj.b = [];
            obj.gradW = [];
            obj.gradB = [];
        end
    end
    
    methods (Access = protected)
    end
    
end