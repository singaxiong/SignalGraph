% this function store gradient in the format of the output
function gradOut = PadGradientVariableLength(grad, mask)
[D,T] = size(grad);
[nFr, nSeg] = size(mask);
if nFr == 1
    nFrActual = ones(nFr, nSeg);
else
    nFrActual = gather(sum(mask==0));
end
if sum(nFrActual)~=T
    fprintf('Error: the number of valid frames is not the same as the size of gradient\n');
end

precision = class(gather(grad(1)));
if strcmpi(class(grad(1)), 'gpuArray')
    gradOut = gpuArray.zeros(D, nFr, nSeg, precision);
else
    gradOut = zeros(D, nFr, nSeg, precision);
end

for i=1:nSeg
    idx1 = sum(nFrActual(1:i-1))+1;
    idx2 = nFrActual(i)+idx1-1;
    gradOut(:,1:nFrActual(i),i) = grad(:,idx1:idx2);
    %gradOut(1,nFrActual(i)+1:end,i) = -1e10;
end
end
