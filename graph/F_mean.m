function output = F_mean(input)
T = size(input,2);

mu = mean(input,2);
output = mu;
%output = repmat(mu, 1,T);

end
