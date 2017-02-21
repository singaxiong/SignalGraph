%
% Perform minmax norm
%
function [output, input_max, scale] = AbsMaxNorm(input,max_val)
if nargin==1
    max_val = 1;
end

input_max = max(abs(input(:)));

scale = max_val ./ input_max;
output = bsxfun(@times, input, scale);

end
