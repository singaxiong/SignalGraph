% Concatenate the neighbouring frames vectors to higher dimensional vectors
% Input is Number of dimension x number of frames
function [y] = SumContext(x, context, window_type)
[dim nFr] = size(x);

if nargin<3
    window_type = 'null';
end

if context>1
    half_context = (context-1)/2;
    if 0    % this implementation may be slow
        y = [];
        for i=1:context
            y = [y; x(:,i:end-context+i)];
        end
    else    
        y = sparse(dim,nFr);    % allocate memory first
        context = gather(context);
        
        nonzero_idx = find(sum(x,2)>0);
        
        x2 = x(nonzero_idx,:);   % only compute the nonzero elements
        x2 = [ zeros(length(nonzero_idx),half_context) x2 zeros(length(nonzero_idx),half_context) ];
        y2 = 0;
        for i=1:context
            y2 = y2 + x2(:,i:end-context+i);
        end
        y(nonzero_idx,:) = y2;
    end
else
    y = x;
end

end