% take the covariance matrix of input trajectories

function output = F_logdet(input)

precision = class(gather(input(1)));
if ~strcmpi(precision, 'double')    % we need to use double precision
    input = double(input);
end

[D,M,N] = size(input);
if N==1
    output = log(det(input));
else
    % to be implemented
end

if strcmpi(precision, 'single')
    output = single(output);
end

end