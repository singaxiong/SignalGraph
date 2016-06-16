%
% write_GMM(FID, priori, mean_matrix, cov_matrix, diag_cov)
%
% Write a GMM to the specified file by the file pointer: FID. The format of
% the input should be: 
%   priori: an one-dimension vector containing the prior probability of the
%           clusters. 
%   mean_matrix: an Dimension x N_cluster two-dimension matrix containing
%           the mean vectors. 
%   cov_matrix: an multi-dimension matrix containing the variance
%           vector/matrix of the clusters. 
%   diag_cov: 1-diagonal covariance matrix
%             0-full covariance matrix
%
% Author: Xiao Xiong
% Created: 7 Feb, 2007
% Last Modified: 18 Feb, 2010

function write_GMM(FID, priori, mean_matrix, cov_matrix, diag_cov)

[Dimension, N_cluster] = size(mean_matrix);

% Check the data
if length(priori) ~= N_cluster
    error('Data dimension not correct or compatible!');
end
if diag_cov
    [d1,d2] = size(cov_matrix);
    if d1~=Dimension || d2~=N_cluster
        error('Data dimension not correct or compatible!');
    end
else
    [d1,d2,d3] = size(cov_matrix);
    if d1~=Dimension || d1~=d2 || d3~=N_cluster
        error('Data dimension not correct or compatible!');
    end
end

% Write the data
fprintf(FID,'<NUMMIXES> %d\n', N_cluster);
for i = 1:N_cluster
    fprintf(FID,'<MIXTURE> %d %f\n', i, priori(i));
    if diag_cov
        write_Gaussian(FID, mean_matrix(:,i) ,cov_matrix(:,i));
    else
        write_Gaussian(FID, mean_matrix(:,i) ,cov_matrix(:,:,i));
    end
end
