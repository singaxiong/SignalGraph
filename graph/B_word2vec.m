function [grad_X, grad_W]= B_word2vec(prev_layer, curr_layer, future_layers, skip_grad_X)
W = curr_layer.W;
input = prev_layer{1}.a;

[dim, nFr, nSeg] = size(input);
if nSeg>1
    input = reshape(input, dim, nFr*nSeg);
end
[m,n] = size(W);

context = dim/n;

if curr_layer.update==0
    grad_W = [];
end

future_grad = GetFutureGrad(future_layers, curr_layer);

nVec = nFr*nSeg;

if curr_layer.update
    grad_W = sparse(size(W,1),size(W,2));
    if 1
        curr_input = reshape(input, n, context * nVec);
        future_grad2 = reshape(future_grad, m, context*nVec);
        nonzero_col= sum(curr_input,2)>0;
        input_nonzero = full(curr_input(nonzero_col,:));
        if strcmpi(class(W),'single')
            input_nonzero = single(input_nonzero);
        end
        grad_W(:,nonzero_col) = double(future_grad2 * input_nonzero');
    else
        for i = 1:context
            curr_input = input( (i-1)*n+1 : i*n, :);
            nonzero_col= sum(curr_input,2)>0;
            input_nonzero = full(curr_input(nonzero_col,:));
            if strcmpi(class(W),'single')
                input_nonzero = single(input_nonzero);
            end
            grad_W(:,nonzero_col) = grad_W(:,nonzero_col) + double(future_grad( (i-1)*m+1 : i*m, :) * input_nonzero');
        end
    end
end
if skip_grad_X==0
    for i = 1:context
        grad_X2{i} = W' * future_grad( (i-1)*m+1 : i*m, :);
    end
    grad_X = cell2mat(grad_X2');
    if nSeg>1
        grad_X = reshape(grad_X, size(grad_X,1), nFr, nSeg);
    end
else
    grad_X = 0;
end

end