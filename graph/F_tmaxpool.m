function [output,idx,validFrameMask] = F_tmaxpool(input_layer, curr_layer)
input = input_layer.a;

% the input contains N samples, each of DxT size
[D,T,N] = size(input);
precision = class(gather(input(1)));

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
    validFrameMask = zeros(1,N);
else
    nWindow = length(1:stride:(T-context+1));
    if strcmpi(class(input), 'gpuArray')
        output = gpuArray.zeros(D, length(nWindow), N, precision);
    else
        output = zeros(D, length(nWindow), N, precision);
    end
    idx = output;
    for i=1:nWindow
        offset = (i-1)*stride;
        [output(:,i,:), idx(:,i,:)] = max(input(:,offset+1:offset+context,:), [], 2);
    end
    validFrameMask = zeros(nWindow,N);
end

end