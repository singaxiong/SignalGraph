%
% [mean_vec,variance] = read_Gaussian(FID,dim,diag_var)
%
% This function read a standard Gaussian distribution from the specified
% file. Both diagonal and full covariance matrix are accepted. User need to
% specify the dimension of the mean vector and whether the covariance
% matrix is diagonal or full. An example (full cov matrix) of the file
% should be: 
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
% Last modified: 6 Feb, 2007

function [mean_vec,variance] = read_Gaussian(FID,diag_cov)

% read the mean
tmp = textscan(FID,'%s %n',1); 
if strcmp(tmp{1}{1},'<MEAN>')==0
    fprintf('Incorrect model file format\n');
    return;
end
dim = tmp{2};
fgetl(FID);
mean_vec = read_matrix(FID,1,dim);

% read the variance
line = fgetl(FID);
tmp = textscan(line,'%s %n',1); 
if (strcmp(tmp{1}{1},'<VARIANCE>')==0 && strcmp(tmp{1}{1},'<INVCOVAR>')==0) || tmp{2}~=dim
    fprintf('Incorrect model file format\n');
    return;
end
if diag_cov         % diagonal variance
    variance = read_matrix(FID,1,dim);
else                % full covariance
    if strcmp(tmp{1}{1},'<INVCOVAR>')
        variance = zeros(dim,dim);
        fgetl(FID);
        for i=1:dim
            tmp = fgetl(FID);
            tmp = textscan(tmp,'%f', dim-i+1);
            variance(i,i:end) = tmp{1};
        end
        tmp = triu(variance)';
        variance = variance + tmp - diag(diag(variance));
        variance = inv(variance);
    else
        variance = read_matrix(FID,dim,dim);
    end
end
tmp = fgetl(FID);
if length(tmp)==0
    tmp = fgetl(FID);
end
if tmp == -1
    return;
end
if length(regexp(tmp, '<GCONST>')) ==0
    fseek(FID, -length(tmp)-1, 'cof');
end


