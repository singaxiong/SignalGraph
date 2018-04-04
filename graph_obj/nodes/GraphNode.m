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
        variableLength = 0;     % indicate whether the trajectories in current minibatch has different lengths
        grad = [];  % gradient of data
        dim = [1 1 1 1];   % feature dimension of the output matrix and input matrix
        rho = [];   % the statistics for L1 norm, reserved for future
        
        % configurations
        skip = 0;       % whether to skip this node in both forward and backward pass. Useful if we only want to evaluate part of the network
        skipBP = 0;     % whether to skip this node in backward pass. Set to 1 for any non-updatable node, on whose parents include no updatable node
        skipGrad = 0;   % whether to skip the computation of grad in backward pass. Set to 1 for the first updatable node in the graph
        L1target = 0;   % target of L1 norm, reserved for future
        L1weight = 0;   % weight of L1 norm, reserved for future
    end
    
    methods
        function obj = GraphNode(name,dimOut)
            obj.name = name;
            if length(dimOut)==1
                obj.dim(1:2) = [dimOut 1];
            else
                obj.dim(1:2) = dimOut;
            end
        end
        function obj = preprocessingForward(obj, prev_layers)
            obj = getValidFrameMask(obj, prev_layers);
        end
        function obj = forward(obj, prev_layers)
            if obj.L1weight>0
                obj.rho = mean(obj.a,2);
            end
        end
        function obj = backward(obj, prev_layers, future_layers)
        end
        
        function obj = cleanUp(obj)
            obj.a = [];
            obj.mask = [];
            obj.grad = [];
            obj.rho = [];
        end
        
        function obj = getValidFrameMask(obj, prev_layers)
            hasMask = zeros(length(prev_layers),1);
            for i=1:length(prev_layers)
                if isprop(prev_layers{i}, 'mask')  && ~isempty(prev_layers{i}.mask)
                    hasMask(i) = 1;
                end
            end
            if sum(hasMask)>0       % if there is already the mask, use it
                idx = find(hasMask==1);
                obj.mask = prev_layers{idx(1)}.mask;        % use the first mask
                obj.variableLength = prev_layers{idx(1)}.variableLength;
            else                    % otherwise, generate it from the data
                obj = obj.CheckTrajectoryLength(prev_layers{1}.a);
            end
        end
        
        % If a trajectory is shorter than others, as specified by the mask, we set
        % its first dimension a defined number. There are two usage of this
        % function:
        %   1. pad the output trajectory of a layer with a big negative
        %   number, e.g. -1e10 to notify
        %   other layers the trajectory is shorter than others in the minibatch.
        %   this is often applied to hidden activations of CNN and LSTM.
        %   2. pad the input trajectory of a layer with 0, so the computation will
        %   carry out for all trajectories as if they have the same length, but do
        %   not affect the results.
        %
        function output = PadShortTrajectory(obj, input, padnumber)
            [~, N] = size(obj.mask);
            [D1,D2,T,N] = size(input);
            input = reshape(input, D1*D2, T, N);
            
            output = input;
            if obj.variableLength==0  % all have the same length
                output = reshape(output, D1, D2, T, N);
                return;
            end
            if strcmpi(padnumber, 'last')   % pad the last frame
                % find the index of the last valide frame in all sequences
                %     delta = mask(2:end,:) - mask(1:end-1,:);
                %     [max_delta, max_idx] = max(delta);
                %     max_idx(max_delta==0) = T;
                max_idx = gather(obj.GetLastValidFrameIndex());
                % pad the last valid frame to the invalide frames
                if 0    % 1 direct implementation
                    for i=1:N
                        output(:,max_idx(i)+1:end,i) = repmat(input(:,max_idx(i),i), 1, T-max_idx(i));
                    end
                else    % 2 index based implementation
                    fr_idx = reshape(1:T*N, T,N); %repmat( (1:T)', 1, N );
                    for i=1:N
                        fr_idx(max_idx(i)+1:end,i) = fr_idx(max_idx(i),i);
                    end
                    output = input(:,fr_idx);
                end
                
                % older implementation
                %     for i=1:N
                %         idx = gather(find(mask(:,i)==1,1));
                %         if ~isempty(idx)
                %             output(:,idx:end,i) = repmat(input(:,idx-1,i), 1, T-idx+1);
                %             idx2(i) = idx;
                %         end
                %     end
                
            elseif padnumber==0
                
                % simple implementation
                output(:,obj.mask==1) = padnumber;
                
                % more complicated implementation
                %     idx = find(reshape(mask, 1, T*N)==1);
                %     if ~isempty(idx)
                %         output = reshape(output, D,T*N);
                %         output(:,idx) = padnumber;
                %         output = reshape(output, D,T,N);
                %     end
            else
                tmp = squeeze(output(1,:,:));
                output(1,:,:) = tmp .* (1-obj.mask) + padnumber*obj.mask;
            end
            output = reshape(output, D1, D2, T, N);
        end
        
        % Extract the valid trajectories from the data
        function [data2] = ExtractVariableLengthTrajectory(data)
            [D1,D2,D3,N] = size(data);
            for i=1:N
                data2{i} = data(:,:,obj.mask(:,i)==0,i);
            end
        end
        
        function data = AllocateMemoryLike(obj, sizes, sample)
            data = zeros(sizes, 'like', sample);
        end
        function data = AllocateMemory(obj, sizes, precision, useGPU)
            if useGPU
                data = gpuArray.zeros(sizes, precision);
            else
                data = zeros(sizes, precision);
            end
        end
    end
    
    methods (Access = protected)
        function future_grad = GetFutureGrad(obj,future_layers)
            for i=1:length(future_layers)
                tmp_future_layer = future_layers{i};
                if tmp_future_layer.skipBP; continue; end
                
                idx = find( tmp_future_layer.prev == -obj.next(i) );   % find the index of current layer in the future layer's prev list
                tmp_grad = tmp_future_layer.grad{idx};
                
                if exist('future_grad', 'var')
                    future_grad = future_grad + tmp_grad;
                else
                    future_grad = tmp_grad;
                end
            end
        end
        
        % check whether the sequences are of the same length.
        % The sequences are aligned from begining. If some sequence is shorter, it
        % will be padded with a big negative number, i.e. -1e10, in the first dimension.
        function obj = CheckTrajectoryLength(obj, data)
            obj.mask = permute(data(1,1,:,:), [3 4 1 2]) == -1e10;
            obj.variableLength = sum(obj.mask(:));
        end
        
        % given a mask of valid frames in sentences, return the an array of index of last
        % valid frames in each sentence.
        %
        function last_idx = GetLastValidFrameIndex(obj)
            T = size(obj.mask,1);
            delta = obj.mask(2:end,:) - obj.mask(1:end-1,:);
            [max_delta, last_idx] = max(delta);
            last_idx(max_delta==0) = T;
            
        end

    end
end