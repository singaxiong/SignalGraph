%
% [priori, mean_matrix, cov_matrix] = read_GMM(FID, dim, N_cluster, diag_cov)
%
% Read a GMM from the specified file by the file pointer: FID. The format of
% the input should be: 
%   FID: the file pointer
%   dim: the number of dimensions
%   N_cluster: number of cluters in the GMM
%   diag_cov: 1-diagonal covariance matrix
%             0-full covariance matrix
%
% Author: Xiao Xiong
% Created: 7 Feb, 2007
% Last Modified: 7 Feb, 2007

function [priori, mean_matrix, cov_matrix] = read_GMM(FID, diag_cov)

tmp = textscan(FID,'%s %n',1);
N_cluster = tmp{2};
fgetl(FID);

for i = 1:N_cluster
    tmp = textscan(FID,'%s %n %n',1);
    priori(i) = tmp{3};
    if diag_cov
        [mean_matrix(:,i),cov_matrix(:,i)] = read_Gaussian(FID,diag_cov);
    else
        [mean_matrix(:,i),cov_matrix(:,:,i)] = read_Gaussian(FID,diag_cov);
    end
end