% Concatenate the neighbouring frames vectors to higher dimensional vectors
% Input is Number of dimension x number of frames
function [y] = ExpandContext_v2(x, context, window_type)
[dim, nFr, nSeg] = size(x);

if nargin<3
    window_type = 'null';
end

context_size = length(context);

if context_size > 1
    if 0
        if strcmp(class(x), 'gpuArray')
            y = gpuArray.zeros(dim*context_size,nFr);
        else
            y = zeros(dim*context_size,nFr);    % allocate memory first
        end
        context = gather(context);
        for i=1:context_size
            idx = min(nFr,max(1, (1:nFr) + context(i)));
            y((i-1)*dim+1:i*dim,:) = x(:,idx);
        end
    else
        context = gather(context);
%         all_idx3 = zeros(context_size,nFr);
        for i=1:context_size
            all_idx{i} = min(nFr,max(1, (1:nFr) + context(i)));
%             all_idx3(i,:) = min(nFr,max(1, (1:nFr) + context(i)));
        end
        all_idx2 = cell2mat(all_idx');
        y_tmp = x(:,all_idx2,:);
        y = reshape(y_tmp, dim*context_size,nFr, nSeg);
    end
else
    y = x;
end

switch window_type
    case 'null'
        window = [];
    case 'Hamming'
        window = hamming(context);
        window = repmat(window, 1, dim);
        window = reshape(window', dim*context, 1);
        window = diag(window);
        y = window * y;
    otherwise
        fprintf('Unkown window type\n');
        return;
end

end