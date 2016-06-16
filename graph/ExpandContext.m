% Concatenate the neighbouring frames vectors to higher dimensional vectors
% Input is Number of dimension x number of frames
function [y] = ExpandContext(x, context, window_type)
[dim nFr] = size(x);

if nargin<3
    window_type = 'null';
end

if context>1
    half_context = (context-1)/2;
    idx = [ones(1,half_context) 1:nFr ones(1,half_context)*nFr];
    x = x(:,idx);
    %x = [ repmat(x(:,1),1,half_context) x repmat(x(:,end), 1, half_context) ];
    if 0    % this implementation may be slow
        y = [];
        for i=1:context
            y = [y; x(:,i:end-context+i)];
        end
    else    
        if strcmp(class(x), 'gpuArray')
            y = gpuArray.zeros(dim*context,nFr);
        else
            y = zeros(dim*context,nFr);    % allocate memory first
        end
        context = gather(context);
        for i=1:context
            y((i-1)*dim+1:i*dim,:) = x(:,i:end-context+i);
        end
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