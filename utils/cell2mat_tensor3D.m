% concatenate cell array of 3D tensors
% assume that the cells are MxNxP_i tensors where M and N are shared by all
% cells, while P_i can be different.
% Generate a 3D tensor of MxNxsum(P_i).
%
function output = cell2mat_tensor3D(data, padnumber)
if nargin<2
    padnumber = 0;
end

precision = 'single';
if ~isempty(data) && ~isempty(data{1})
    precision = class(data{1}(1));
end

nCell = length(data);
for i=1:nCell
    [M(i),N(i),P(i)] = size(data{i});
    if M(i)*N(i)==0
        P(i)=0;
    end
end

if sum(abs(N-N(1)))==0  % if all cells have the same number of columns
    N = N(1);
else        % if some cells have less columns than others, zero pad them
    Nmax = max(N);
    for j=1:nCell
        data{j}(:,N(j)+1:Nmax,:) = padnumber;
    end
    N = max(N);
end

if sum(abs(P-1))==0     % all are matrices, not tensors
    output = cell2mat(data);
    output = reshape(output, max(M),N,nCell);
else
    output = zeros(max(M),N, sum(P), precision);
    for j=1:nCell
        idx1 = sum(P(1:j-1))+1;
        idx2 = sum(P(1:j));
        if idx1>idx2; continue; end
        [d1,d2,d3] = size(data{j});
        [n1,n2,n3] = size(output(:,:, idx1:idx2 ));
        
        if d1~=n1 || d2~=n2 || d3~=n3 %M(j)~=max(M) || idx2-idx1+1 ~= P(j)
            msg = sprintf('Warning: data{j} has a dimension of %d - %d - %d, while output(:,:,idx1:idx2) has a dimension of %d - %d - %d, %s\n', d1,d2,d3,n1,n2,n3, datestr(now));
            my_append('cell2mat_tensor3D_log.txt', msg);
            continue;
        end
        output(:,:, idx1:idx2 ) = data{j};
    end
end

end
