classdef LSTMNode < GraphNodeUpdatable
    properties
        useHidden = 1;
        usePastState = 1;
        usePastStateAsFeature = 0;
        ft;
        it;
        ot;
        Ct_raw;
        Ct;
        Ct0;
        ht0;
    end
    
    methods
        function obj = LSTMNode(dimOut)
            obj = obj@GraphNodeUpdatable('LSTM', dimOut);
        end
        
        function obj = initialize(obj,useGaussInit, useNegbiasInit, r)
            input_size = obj.dim(3); hidden_size = obj.dim(1);
            input_and_recurrent_size = hidden_size*(1+obj.usePastStateAsFeature)+input_size;
            if isempty(obj.W) || sum(abs(size(obj.W)-[hidden_size*4 input_and_recurrent_size]))>0
                % the weight matrix of LSTM is organized as follows:
                % W = [ W_cf W_hf W_xf;
                %       W_cc W_hc W_xc;
                %       W_ci W_hi W_xi;
                %       W_co W_ho W_xo;]
                % where W_cf is the weights that connects past cell states to
                % forget gates, W_hc is the weights between past hidden activation
                % to candidate states, W_xi is the weights beween input features to
                % input gates. Other weights are similarly defined.
                obj.W = randn(4 * hidden_size, input_and_recurrent_size) / sqrt(input_and_recurrent_size);
                if obj.usePastStateAsFeature
                    % we need to make W_cc zero as there is no connection between past
                    % cell states to candidate cell states.
                    obj.W( hidden_size+1:hidden_size*2, 1:hidden_size ) = 0;
                end
            end
            if isempty(obj.b) || sum(abs(size(obj.b,1)-hidden_size*4))>0
                obj.b = zeros(hidden_size*4,1);
            end
        end
        
        function obj = forward(obj,prev_layers)
            obj = obj.preprocessingForward(prev_layers);

            input_layer = prev_layers{1};
            input = input_layer.a;
            
            % We introduce a vector at = [ft; Ct_raw; it; ot], and zt is the vector of at
            % before the sigmoid or tanh activation functions. We also introduce matrix
            % W = [Wc Wh Wx]; where Wc = [W_cf; W_cc; W_ci; W_co] and so on.
            % Let st = [Ct-1 ht-1 input]. Then, the LSTM will first compute
            % zt = W * st.
            
            % Wc and Wh are 4N x N matrices, where N is the number of cells in the
            % layer. Wx is a 4N x dim matrix, where dim is the number of inputs.
            
            % input are multiple trajectories, usually of different lengths. We need to
            % first find the number of effective frames
            
            [dim, D2, nFr, nSeg] = size(input);
            assert(D2==1, sprintf('%s:forward, Error: dimension 2 (%d) is not equal to 1', obj.name, D2));
            input = squeeze(input);
            if nSeg>1
                if obj.variableLength; input = obj.PadShortTrajectory(input, 0); end
            end
            
            input = permute(input, [1 3 2]);
            precision = class(gather(input(1)));
            
            nCell = obj.dim(1);  % number of LSTM cells in the layer
            
            if obj.usePastStateAsFeature
                Wc = obj.W(:,1:nCell);           % weights of past cell states. Note that the second block of Wc corresponding to Ct_raw is zero.
                Wh = obj.W(:,nCell+1:nCell*2);   % weights of past hidden activations
                Wx = obj.W(:,nCell*2+1:end);     % weights of current input vector
            else
                Wh = obj.W(:,1:nCell);   % weights of past hidden activations
                Wx = obj.W(:,nCell+1:end);     % weights of current input vector
            end
            b = obj.b;     % bias
            
            % initialize LSTM state vector and hidden layer output
            if IsInGPU(input)==0
                Ct0 = ones(nCell,nSeg, precision);   % initial cell states
                ht0 = ones(nCell,nSeg, precision);   % initial hidden layer output
                % allocate memory for gates and states
                %     ft = zeros(nCell, nSeg, nFr, precision);     % forget gates
                %     it = zeros(nCell, nSeg, nFr, precision);     % input gates
                %     ot = zeros(nCell, nSeg, nFr, precision);     % output gates
                %     Ct_raw = zeros(nCell, nSeg, nFr, precision); % candidate cell states
                %     Ct = zeros(nCell, nSeg, nFr, precision);     % cell states
                %     ht = zeros(nCell, nSeg, nFr, precision);     % hidde layer output, i.e. the output of the LSTM layer
                Gates = zeros(nCell*7, nSeg, nFr, precision);
            else
                Ct0 = gpuArray.ones(nCell,nSeg, precision);   % initial cell states
                ht0 = gpuArray.ones(nCell,nSeg, precision);   % initial hidden layer output
                % allocate memory for gates and states
                %     ft = gpuArray.zeros(nCell, nSeg, nFr, precision);     % forget gates
                %     it = gpuArray.zeros(nCell, nSeg, nFr, precision);     % input gates
                %     ot = gpuArray.zeros(nCell, nSeg, nFr, precision);     % output gates
                %     Ct_raw = gpuArray.zeros(nCell, nSeg, nFr, precision); % candidate cell states
                %     Ct = gpuArray.zeros(nCell, nSeg, nFr, precision);     % cell states
                %     ht = gpuArray.zeros(nCell, nSeg, nFr, precision);     % hidde layer output, i.e. the output of the LSTM layer
                Gates = gpuArray.zeros(nCell*7, nSeg, nFr, precision);
            end
            
            % batch transform the input features for fast speed
            AffineLayer = AffineNode(size(Wx,1));
            AffineLayer.W = Wx;
            AffineLayer.b = b;
            prevLayers{1}.a = input;
            AffineLayer = AffineLayer.forward(prevLayers);
            z_from_inputs = AffineLayer.a;
            
            zt_curr_idx = zeros(nCell*4,1);
            zt_curr_idx(nCell*1+1:nCell*2,:) = 1;
            zt_curr_idx = logical(zt_curr_idx);
            
            for i=1:nFr
                if i==1     % for the first frame, use default values for past state and hidden values.
                    Ct_past = Ct0;
                    ht_past = ht0;
                else
                    Ct_past = Ct_curr;
                    ht_past = ht_curr;
                end
                
                zt = z_from_inputs(:,:, i);
                if obj.usePastStateAsFeature
                    zt = zt + Wc * Ct_past;
                end
                if obj.useHidden
                    zt = zt + Wh * ht_past;
                end
                
                % extract the elements of zt to compute the gates
                zt_sig = sigmoid(zt);
                % Ct_raw_curr    = tanh(     zt(nCell+1:nCell*2,:)   );
                Ct_raw_curr    = tanh(     zt(zt_curr_idx,:)   );   % slightly faster
                
                if 0
                    ft_curr = zt_sig(1:nCell,:);
                    it_curr = zt_sig(nCell*2+1:nCell*3,:);
                    ot_curr = zt_sig(nCell*3+1:nCell*4,:);
                else    % 5% faster
                    zt_sig2 = permute(reshape(zt_sig, nCell, 4, nSeg), [1 3 2]);
                    ft_curr = zt_sig2(:,:,1);
                    it_curr = zt_sig2(:,:,3);
                    ot_curr = zt_sig2(:,:,4);
                end
                
                % compute the states of the cells, which is a weighted sum of the
                % current state and past state.
                if obj.usePastState
                    Ct_curr = Ct_raw_curr .* it_curr + Ct_past .* ft_curr;
                else
                    Ct_curr = Ct_raw_curr .* it_curr;
                end
                
                % compute the output
                ht_curr = tanh(Ct_curr) .* ot_curr;
                
                Gates(:,:,i) = [zt_sig; Ct_raw_curr; Ct_curr; ht_curr];
            end
            
            ft = Gates(1:nCell,:,:);
            it = Gates(nCell*2+1:nCell*3,:,:);
            ot = Gates(nCell*3+1:nCell*4,:,:);
            Ct_raw = Gates(nCell*4+1:nCell*5,:,:);
            Ct = Gates(nCell*5+1:nCell*6,:,:);
            ht = Gates(nCell*6+1:nCell*7,:,:);
            
            if nSeg>1 && obj.variableLength
                ht = permute(obj.PadShortTrajectory(permute(ht,[1 3 2]), -1e10), [1 3 2]);
                ft = permute(obj.PadShortTrajectory(permute(ft,[1 3 2]), 0), [1 3 2]);
                it = permute(obj.PadShortTrajectory(permute(it,[1 3 2]), 0), [1 3 2]);
                ot = permute(obj.PadShortTrajectory(permute(ot,[1 3 2]), 0), [1 3 2]);
                Ct_raw = permute(obj.PadShortTrajectory(permute(Ct_raw,[1 3 2]), 0), [1 3 2]);
                Ct = permute(obj.PadShortTrajectory(permute(Ct,[1 3 2]), 0), [1 3 2]);
            end
            
            obj.ft = ft;
            obj.it = it;
            obj.ot = ot;
            obj.Ct_raw = Ct_raw;
            obj.Ct = Ct;
            obj.a = permute(ht, [1 4 3 2]);
            obj.Ct0 = Ct0;
            obj.ht0 = ht0;
            
            obj = forward@GraphNodeUpdatable(obj, prev_layers);
        end
        
        function obj = backward(obj,prev_layers, future_layers)
            input = prev_layers{1}.a;
            
            future_grad = obj.GetFutureGrad(future_layers);
            
            W = obj.W;
            precision = class(gather(input(1)));
            
            [dim, D2, nFr, nSeg] = size(input);
            assert(D2==1, sprintf('%s:forward, Error: dimension 2 (%d) is not equal to 1', obj.name, D2));
            input = squeeze(input);
            if nSeg>1 && obj.variableLength
                future_grad = obj.PadShortTrajectory(future_grad, 0);
                input = obj.PadShortTrajectory(input, 0);
            end
            
            input = permute(input, [1 3 2]);
            
            nCell = obj.dim(1);  % number of LSTM cells in the layer
            
            grad_ht_cost = squeeze(future_grad);
            grad_ht_cost = permute(grad_ht_cost,[1 3 2]);
            
            ft = obj.ft;
            it = obj.it;
            ot = obj.ot;
            Ct_raw = obj.Ct_raw;
            Ct = obj.Ct;
            ht = squeeze(obj.a);
            ht = permute(ht, [1 3 2]);
            Ct0 = obj.Ct0;
            ht0 = obj.ht0;
            
            % allocate memory for the gradients of gates and states
            if IsInGPU(input) == 0
                if obj.skipGrad==0; grad_xt = zeros(dim, nSeg, nFr, precision); end
                grad_Ct_future_curr = zeros(nCell, nSeg, precision);     % cell states
                grad_ht_future_curr = zeros(nCell, nSeg, precision);     % hidde layer output, i.e. the output of the LSTM layer
                grad_zt = zeros(nCell*4,nSeg, nFr, precision);
            else
                if obj.skipGrad==0; grad_xt = gpuArray.zeros(dim, nSeg, nFr, precision); end
                grad_Ct_future_curr = gpuArray.zeros(nCell, nSeg, precision);     % cell states
                grad_ht_future_curr = gpuArray.zeros(nCell, nSeg, precision);     % hidde layer output, i.e. the output of the LSTM layer
                grad_zt = gpuArray.zeros(nCell*4,nSeg, nFr, precision);
            end
            
            Ct_raw_prod = 1 - Ct_raw.*Ct_raw;
            tanh_Ct_all = tanh(Ct);
            termFromTanhCtAndOt = (1-tanh_Ct_all.*tanh_Ct_all) .* ot;
            Gates = [ft; it; ot];
            GatesTerm =  Gates .* (1-Gates);
            W2 = W';
            W2 = [W2(:,1:nCell) W2(:,2*nCell+1:end) W2(:,nCell+1:2*nCell)];
            Ct_CtRaw_tanCtAll = [Ct(:,:,1:end-1); Ct_raw(:,:,2:end); tanh_Ct_all(:,:,2:end)];
            
            % we only need the current value of the grad of the gates, we don't need to
            % store them
            
            for t = nFr:-1:1
                % compute the gradient of ht and Ct that requires gradients from
                % future. At frame nFr, the future gradients are initialized to 0.
                if obj.useHidden
                    grad_ht_curr = grad_ht_future_curr + grad_ht_cost(:,:,t);
                else
                    grad_ht_curr = grad_ht_cost(:,:,t);
                end
                grad_Ct_curr = grad_Ct_future_curr + grad_ht_curr .* termFromTanhCtAndOt(:,:,t);
                
                % compute the gradient of gates and candidate states
                if obj.usePastState
                    if t==1
                    else
                        grad_Ct_future_curr = grad_Ct_curr .* ft(:,:,t);
                    end
                end
                grad_Ct_raw_curr = grad_Ct_curr .* it(:,:,t);
                
                if t==1
                    grad_gates = [grad_Ct_curr; grad_Ct_curr; grad_ht_curr] .* [Ct0; Ct_raw(:,:,t); tanh_Ct_all(:,:,t)];
                else
                    grad_gates = [grad_Ct_curr; grad_Ct_curr; grad_ht_curr] .* Ct_CtRaw_tanCtAll(:,:,t-1); %[Ct(:,:,t-1); Ct_raw(:,:,t); tanh_Ct_all(:,:,t)];
                end
                
                % compute the gradient of the gates before the activation function.
                
                grad_zCt_raw = grad_Ct_raw_curr .* Ct_raw_prod(:,:,t);    % batch processing on Ct_raw
                grad_zgates = grad_gates .* GatesTerm(:,:,t);   % apply operation independent of loop in batch
                grad_zt_curr = [grad_zgates; grad_zCt_raw];     % avoid complicated memory operation
                grad_zt(:,:,t) = grad_zt_curr;
                
                % compute the gradient of the W, b, and past hidden, past state, and x
                grad_yt = W2 * grad_zt_curr;    % use re-organized W
                
                if obj.skipGrad==0
                    if  obj.usePastStateAsFeature
                        grad_xt(:,:,t) = grad_yt(nCell*2+1:end,:);
                    else
                        grad_xt(:,:,t) = grad_yt(nCell+1:end,:);    % if the grad w.r.t. input data is not necessary (i.e. no trainable parameters before this LSTM layer, we don't need to compute grad_xt
                    end
                end
                if t==1
                else
                    if obj.usePastStateAsFeature
                        grad_Ct_future_curr = grad_Ct_future_curr + grad_yt(1:nCell,:);
                        grad_ht_future_curr = grad_yt(nCell+1:nCell*2,:);
                    else
                        grad_ht_future_curr = grad_yt(1:nCell,:);
                    end
                end
            end
            grad_zt = [grad_zt(1:nCell,:,:); grad_zt(end-nCell+1:end,:,:); grad_zt(nCell+1:end-nCell,:,:)];   % re-organize grad_zt
            
            if obj.usePastStateAsFeature
                tmpMat = [Ct(:,:,1:nFr-1); ht(:,:,1:nFr-1)*obj.useHidden; input(:,:,2:nFr)];
            else
                tmpMat = [ht(:,:,1:nFr-1)*obj.useHidden; input(:,:,2:nFr)];
            end
            tmpMat = reshape(tmpMat, size(tmpMat,1), nSeg*(nFr-1));
            tmpMat2 = reshape(grad_zt(:,:,2:nFr), nCell*4, nSeg*(nFr-1));
            obj.gradW = tmpMat2 * tmpMat';
            if obj.usePastStateAsFeature
                obj.gradW = obj.gradW + grad_zt(:,:,1) * [Ct0; ht0*obj.useHidden; input(:,:,1)]';
            else
                obj.gradW = obj.gradW + grad_zt(:,:,1) * [ht0*obj.useHidden; input(:,:,1)]';
            end
            
            obj.gradB = sum(sum(grad_zt,3),2);
            if obj.skipGrad==0
                obj.grad{1} = permute(grad_xt,[1 4 3 2]);
            else
                obj.grad{1} = [];
            end
            
            obj = backward@GraphNodeUpdatable(obj, prev_layers, future_layers);
            
        end
        
        function obj = cleanUp(obj)
            obj = cleanUp@GraphNodeUpdatable(obj);
            obj.ft = [];
            obj.it = [];
            obj.ot = [];
            obj.Ct_raw = [];
            obj.Ct = [];
            obj.Ct0 = [];
            obj.ht0 = [];
        end
        
    end
    
end