function layer = initializeParametersDNN_tree(layer, para)
negbias = para.NET.useNegbiasInit;
gauss = para.NET.useGaussInit;

% Initialize parameters randomly based on layer sizes.
nNode = 0;
for i=1:length(layer)
    if strcmpi(layer{i}.name, 'affine') || strcmpi(layer{i}.name, 'input') || strcmpi(layer{i}.name, 'word2vec')
        nNode = nNode + layer{i}.dim(1);
    end
end

r  = sqrt(6) / sqrt(nNode);   % we'll choose weights uniformly from the interval [-r, r]
for i=1:length(layer)
    if strcmpi(layer{i}.name, 'affine') || strcmpi(layer{i}.name, 'tconv')
        if layer{i}.update && para.NET.rmsprop_decay>0 
            layer{i}.gradW_avg_square = 0;
        end
        if length(layer{i}.prev)>1; continue; end   % if there are more than 1 input layer, one of them will provide parameters
        if isfield(layer{i}, 'W')==0 || sum(abs(size(layer{i}.W)-layer{i}.dim))>0
            if gauss
                layer{i}.W = 3/sqrt(layer{i}.dim(2)) * randn(layer{i}.dim);
            else
                layer{i}.W = rand(layer{i}.dim) * 2 * r - r;
            end
        end
        if layer{i}.update && para.NET.rmsprop_decay>0
            layer{i}.gradb_avg_square = 0;
        end
        if isfield(layer{i}, 'b')==0 || sum(abs(size(layer{i}.b,1)-layer{i}.dim(1)))>0
            if negbias
                layer{i}.b = rand(layer{i}.dim(1),1)/5 - 4.1;
            else
                layer{i}.b = zeros(layer{i}.dim(1),1);
            end
        end
    elseif strcmpi(layer{i}.name, 'word2vec')
        trans_size = layer{i}.dim / layer{i}.context;
        if isfield(layer{i}, 'W')==0 || sum(abs(size(layer{i}.W)-trans_size))>0
            if gauss
                layer{i}.W = 3/sqrt(trans_size(2)) * randn(trans_size);
            else
                layer{i}.W = rand(trans_size) * 2 * r - r;
            end
        end
    elseif strcmpi(layer{i}.name, 'LSTM')
        input_size = layer{i}.dim(2); hidden_size = layer{i}.dim(1);
        input_and_recurrent_size = hidden_size*(1+layer{i}.usePastState)+input_size;
        if isfield(layer{i}, 'W')==0 || sum(abs(size(layer{i}.W)-[hidden_size*4 input_and_recurrent_size]))>0
            % the weight matrix of LSTM is organized as follows:
            % W = [ W_cf W_hf W_xf;
            %       W_cc W_hc W_xc;
            %       W_ci W_hi W_xi;
            %       W_co W_ho W_xo;]
            % where W_cf is the weights that connects past cell states to
            % forget gates, W_hc is the weights between past hidden activation
            % to candidate states, W_xi is the weights beween input features to
            % input gates. Other weights are similarly defined.
            layer{i}.W = randn(4 * hidden_size, input_and_recurrent_size) / sqrt(input_and_recurrent_size);
            if layer{i}.usePastState
                % we need to make W_cc zero as there is no connection between past
                % cell states to candidate cell states.
                layer{i}.W( hidden_size+1:hidden_size*2, 1:hidden_size ) = 0;
            end
        end
        if isfield(layer{i}, 'b')==0 || sum(abs(size(layer{i}.b,1)-hidden_size*4))>0
            layer{i}.b = zeros(hidden_size*4,1);
        end
    end
end

if ~isempty(para.NET.WeightTyingSet)
    for i=1:length(para.NET.WeightTyingSet)
        currTyingSet = para.NET.WeightTyingSet{i};
        [dimMismatch, isTranspose] = VerifyTiedLayers(layer(currTyingSet));
        
        baseNode = layer{currTyingSet(1)};
        for j=2:length(currTyingSet)
            if isfield(baseNode, 'W')
                if isTranspose(j)
                    layer{currTyingSet(j)}.W = baseNode.W';
                else
                    layer{currTyingSet(j)}.W = baseNode.W;
                end
            end
            if isfield(baseNode, 'b')   && ~isTranspose(j)  % if the two shared layers are transpose of each other, we don't share the bias
                layer{currTyingSet(j)}.b = baseNode.b;
            end
        end
    end
end

end

