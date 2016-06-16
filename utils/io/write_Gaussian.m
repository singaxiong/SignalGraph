%
% write_Gaussian(FID,mean_vec,variance)
%
% This function write a standard Gaussian distribution into the specified
% file. Both diagonal and full covariance matrix are accepted. In case of
% diagonal covariance matrix, only a variance vector is written. An example
% (full cov matrix) of the file will be: 
%
% ...
% Mean Vector:
% xxx xxx xxx ... xxx       % Mean vector
% Variance Matrix:
% xxx xxx xxx ... xxx       % One vector of covariance matrix
% ...
% xxx xxx xxx ... xxx
% ...
%
% Author: Xiao Xiong
% Created: 6 Feb, 2007
% Last modified: 18 Feb, 2010

function write_Gaussian(FID,mean_vec,variance)

dim = length(mean_vec);
[m,n] = size(variance);

if m*n ~= dim && m*n ~= dim^2  
    error('The dimension of the mean vector and variance matrix are not compatible');
else
    % write the mean vector
    fprintf(FID,'<MEAN> %d\n',dim);
    write_matrix(FID, reshape(mean_vec, 1,dim) );
    
    fprintf(FID,'<VARIANCE> %d\n',dim);
    % write the variance
    if m*n == dim       % diagonal variance
        write_matrix(FID, reshape(variance, 1,dim) );
    else                % full covariance
        write_matrix(FID, variance);
    end
end
