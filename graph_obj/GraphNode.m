% Class for nodes without parameters
classdef GraphNode
    properties
        name = '';
        
        % graph traversal variables
        myIdx = []; % the index of this node in the graph
        prev = -1;  % offset of previous layers w.r.t. current layer
        next = [];  % offset of future layers w.r.t. current layer
        
        % weights, gradients, and activations
        a = [];     % activation
        mask = [];  % mask of whether a particular frame of a sequence is valid
        grad = [];  % gradient of data
        dim = [];   % feature dimension of the output
        rho = [];   % the statistics for L1 norm
        
        % configurations
        skipBP = 0; %whether to skip this node in backward pass
        skipGrad = 0;   % whether to skip the computation of grad in backward pass
        L1target = 0;   % target of L1 norm
        L1weight = 0;   % weight of L1 norm
    end
    
    methods
        function obj = GraphNode(name, myIdx)
            obj.name = name;
            obj.myIdx = myIdx;
        end
        function obj = forward(obj, prev_layers)
            if obj.L1weight>0
                obj.rho = mean(obj.a,2);
            end
        end
        function obj = backward(obj, future_layers, prev_layers)
        end
        function rho = L1cost(obj)

        end
        function obj = cleanUp(obj)
            obj.a = [];
            obj.mask = [];
            obj.grad = [];
        end
    end
    
    methods (Access = protected)
        function future_grad = GetFutureGrad(obj,future_layers)
            for i=1:length(future_layers)
                tmp_future_layer = future_layers{i};
                if tmp_future_layer.skipBP; continue; end
                
                switch lower(tmp_future_layer.name)
                    case {'weighted_average'}
                        if tmp_future_layer.prev(1) == -curr_layer.next
                            tmp_grad = tmp_future_layer.grad_W_raw;
                        else
                            tmp_grad = tmp_future_layer.grad;
                        end
                    case {'inner_product_normalized', 'concatenate', 'cosine', 'spatialcovsplitmask',...
                            'hadamard', 'lda', 'beamforming', 'add', 'matrix_multiply', 'll_gaussian', 'mixture_mse'}
                        idx = tmp_future_layer.prev == -curr_layer.next(i);    % find out which of the future grad cell contains the gradient for current layer.
                        % The prev of the future layer for the current layer mathces the negative of the next of the current layer for the future layer.
                        tmp_grad = tmp_future_layer.grad{idx};
                    case {'affine', 'tconv'}
                        if length(tmp_future_layer.prev)>1
                            idx = tmp_future_layer.prev == -curr_layer.next(i);
                            tmp_grad = tmp_future_layer.grad{idx};
                        else
                            tmp_grad = tmp_future_layer.grad;
                        end
                    otherwise       % for normal future layers, the grad is a matrix, not a cell array.
                        tmp_grad = tmp_future_layer.grad;
                end
                
                if exist('future_grad', 'var')
                    future_grad = future_grad + tmp_grad;
                else
                    future_grad = tmp_grad;
                end
            end
            
            
        end
        
    end
end