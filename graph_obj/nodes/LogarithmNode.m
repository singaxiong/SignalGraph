classdef LogarithmNode < GraphNode
    properties
        const = 1e-2;
    end
    
    methods
        function obj = LogarithmNode(dimOut, const)
            obj = obj@GraphNode('Logarithm',dimOut);
            if nargin>=2
                obj.const = const;
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);
            input = prev_layers{1}.a;
            [D,T,N] = size(input);
            
            if strcmpi(class(gather(input(1))), 'single')
                obj.const = single(obj.const);
            end
            
            if N==1
                obj.a = log(input+obj.const);
            else
                if obj.variableLength
                    input = obj.PadShortTrajectory(input, 0);
                    if sum(input(:)+obj.const<=0)
                        fprintf('error: input to log is negative');
                    end
                    obj.a = log(input+obj.const);
                else
                    obj.a = log(input+obj.const);
                end
            end
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            future_grad = obj.GetFutureGrad(future_layers);
            input = prev_layers{1}.a;
            obj.grad{1} = 1./(input+obj.const).*future_grad;
        end
    end
end