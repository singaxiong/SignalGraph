% If a future layer has multiple inputs, it's grad will be a cell array,
% each cell is for one input. A cell is empty if the corresponding input
% does not need gradient to be propagated.
% This function is used to handle the various scenarios when the future
% layer has multiple inputs.

function future_grad = GetFutureGrad(future_layers, curr_layer)

for i=1:length(future_layers)
    tmp_future_layer = future_layers{i};
    if isfield(tmp_future_layer, 'skipBP') && tmp_future_layer.skipBP
        continue;
    end
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
