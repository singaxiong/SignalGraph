
function grad = B_splice_single_sentence(grad, future_grad, context, mask)
[dim,nFr,nSeg] = size(future_grad);
dim = dim/context;
half_ctx = (context-1)/2;

tmp1 = []; tmp2 = [];
for i=-half_ctx:half_ctx
    curr_future_grad = future_grad( (i+half_ctx)*dim+1 : (i+half_ctx+1)*dim, :,:);
    if i<0
        grad(:,1:end+i,:) = grad(:,1:end+i,:) + curr_future_grad(:, -i+1:end,:);
        tmp1 = [tmp1 curr_future_grad(:, 1:-i,:)];
    elseif i>0
        grad(:,i+1:end,:) = grad(:,i+1:end,:) + curr_future_grad(:, 1:end-i,:);
        tmp2 = [tmp2 curr_future_grad(:, end-i+1:end,:)];
    else
        grad = grad + curr_future_grad;
    end
end
grad(:,1,:) = grad(:,1,:) + sum(tmp1,2);
grad(:,end,:) = grad(:,end,:) + sum(tmp2,2);

end