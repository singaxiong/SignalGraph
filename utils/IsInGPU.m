% check whether a variable is in GPU or not
function output = IsInGPU(x)
if strcmpi(class(x), 'gpuArray')
    output = 1;
else
    output = 0;
end
end
