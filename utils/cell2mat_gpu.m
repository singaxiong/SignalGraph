% concatenate several matrixes in GPU memory to a big matrix
function output = cell2mat_gpu(data)
precision = 'single';
if ~isempty(data) && ~isempty(data{1})
    isGPUArray = strcmpi(class(data{1}(1)), 'gpuArray');
    precision = class(gather(data{1}(1)));
end

nCell = length(data);
for i=1:nCell
    [M,N(i)] = size(data{i});
end

if isGPUArray
    output = gpuArray.zeros(M,sum(N), precision);
else
    output = zeros(M,sum(N), precision);
end
for j=1:nCell
    idx1 = sum(N(1:j-1))+1;
    idx2 = sum(N(1:j));
    if idx1>idx2; continue; end
    output(:,idx1:idx2 ) = data{j};
end

end
