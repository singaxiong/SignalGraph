function [output,idx] = F_tmaxpool(input, curr_layer)

% the input contains N samples, each of DxT size
[D,T,N] = size(input);

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

if context==0 || stride==0  % global pooling
    [output,idx] = max(input,[], 2);
else
    nWindow = length(1:stride:(T-context+1));
    if strcmpi(class(input), 'gpuArray')
        output = gpuArray.zeros(D, length(nWindow), N);
    else
        output = zeros(D, length(nWindow), N);
    end
    idx = output;
    for i=1:nWindow
        offset = (i-1)*stride;
        [output(:,i,:), idx(:,i,:)] = max(input(:,offset+1:offset+context,:), [], 2);
    end
end

end