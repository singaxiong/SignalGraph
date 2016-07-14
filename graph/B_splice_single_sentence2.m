
function grad = B_splice_single_sentence2(future_grad, context, mask)
[dim,nFr,nSeg] = size(future_grad);
dim = dim/context;
half_ctx = (context-1)/2;

future_grad2 = reshape(future_grad, dim, context, nFr, nSeg);
future_grad2(:,:,end+1,:) = 0;

for i=-half_ctx:half_ctx
    all_idx{i+half_ctx+1} = min(nFr,max(1, (1:nFr) + i));
end
all_idx2 = cell2mat(all_idx');

future_grad2 = all_idx2;
future_grad2(:,end+1) = 0;

grad_idx = zeros(nFr, context);
grad_idx1 = [];
grad_idxend = [];
for i=-half_ctx:half_ctx
    offset = (nFr+1)*(i+half_ctx);
    % curr_future_grad = future_grad( (i+half_ctx)*dim+1 : (i+half_ctx+1)*dim, :,:);
    if i<0
        grad_idx(1:nFr+i, i+half_ctx+1) = (-i+1:nFr) + offset;
        %grad(:,1:end+i,:) = grad(:,1:end+i,:) + curr_future_grad(:, -i+1:end,:);
        grad_idx1 = [grad_idx1 (1:-i)+offset];
    elseif i>0
        grad_idx(i+1:nFr, i+half_ctx+1) = (1:nFr-i) + offset;
        
        %grad(:,i+1:end,:) = grad(:,i+1:end,:) + curr_future_grad(:, 1:end-i,:);
        %tmp2 = [tmp2 curr_future_grad(:, end-i+1:end,:)];
        grad_idxend = [grad_idxend (nFr-i+1:nFr)+offset];
    else
        grad_idx(1:nFr, i+half_ctx+1) = (1:nFr) + offset;
        %grad = grad + curr_future_grad;
    end
end
grad_idx(grad_idx==0) = nFr*context+1; 

% future_grad2 = permute(future_grad2, [1 4 2 3]); 
future_grad2 = reshape(future_grad2, dim*nSeg, context*(nFr+1));
grad = future_grad2(:,grad_idx);
grad = reshape(grad, dim, context, nFr);
grad = squeeze(sum(grad,2));
grad = permute(grad, [1 3 2]);


grad = future_grad2(grad_idx');

% grad(:,1,:) = grad(:,1,:) + sum(tmp1,2);
% grad(:,end,:) = grad(:,end,:) + sum(tmp2,2);


end