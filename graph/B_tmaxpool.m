function grad = B_tmaxpool(prev_layer, curr_layer, future_layers)
input = prev_layer{1}.a;
[D,T,N] = size(input);
max_idx = curr_layer.idx;

if isfield(curr_layer, 'context')
    context = curr_layer.context;
else
    context = 0;
end
if isfield(curr_layer, 'stride')
    stride = curr_layer.stride;
else
    stride = 0;
end

future_grad = GetFutureGrad(future_layers, curr_layer);

if strcmpi(class(future_grad), 'gpuArray')
    grad = gpuArray.zeros(D,T,N);
else
    grad = zeros(D,T,N);
end

if context==0 || stride==0  % global pooling
    if 0
        for n=1:N
            for d = 1:D
                grad(d,max_idx(d,1,n),n) = future_grad(d,1,n);
            end
        end
    elseif 0
        for n = 1:N
            % for new Matlab version, we can use sub2idx.
            % idx2 = sub2idx(size(grad(:,:,1)), 1:D, max_idx(:,1,n));
            offset = (max_idx(:,1,n)-1)*D + D*T*(n-1);
            idx = offset+ [1:D]';
            grad(idx) = future_grad(:,1,n);
        end
    else
        offset = bsxfun(@plus, (squeeze(max_idx)-1)*D, D*T*(0:(N-1)) );
        offset = reshape(offset, numel(future_grad),1);
        idx = offset+ repmat((1:D)', N,1);
        grad(idx) = future_grad;
    end
else
    if 0
        for i=1:size(max_idx,2)
            offset = (i-1)*stride;
            for n=1:N
                for d = 1:D
                    idx = max_idx(d,i,n) + offset;
                    grad(d,idx,n) = future_grad(d,i,n);
                end
            end
        end
    else
        T2 = size(max_idx,2);
        max_idx = reshape(permute(max_idx, [1 3 2]), D*N,T2);
        future_grad = reshape(permute(future_grad, [1 3 2]), D*N,T2);
        grad = reshape(permute(grad, [1 3 2]), D*N,T);

        if 0
            for i=1:T2
                offset = (i-1)*stride;
                offset2 = (offset+max_idx(:,i)-1)*D*N;
                idx = offset2 + [1:(D*N)]';
                grad(idx) = future_grad(:,i);
            end
        else
            offset = (0:(T2-1))*stride;
            offset2 = bsxfun(@plus, (max_idx-1)*D*N, offset*D*N);
            offset2 = reshape(offset2, numel(offset2),1);
            idx = offset2 + repmat([1:(D*N)]', T2,1);
            grad(idx) = future_grad;
        end
        grad = permute(reshape(grad, D,N,T), [1 3 2]);
    end
end

end