% take the covariance matrix of input trajectories

function output = F_cov(input)

[D,M,N] = size(input);
if N==1
    %output = input*input' / M;
    output = cov(input');
else
    % to be implemented
end

end