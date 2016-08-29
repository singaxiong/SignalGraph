function [layer, update, total_weight_norm] = DNN_update(layer, para, update, itr, learning_rate)
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
