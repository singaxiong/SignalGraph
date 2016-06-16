%
% write_matrix(FID,data)
%
% This function write a matrix to the specified file. Both two dimensional
% matrix and one dimensional vector are accepted. 
%
% Author: Xiao Xiong
% Created: 6 Feb, 2007
% Last modified: 6 Feb, 2007

function write_matrix(FID,data, exponential)

if nargin<3
    exponential =0;
end
if ndims(data) > 2
    error('The dimension of the data cannot be larger than 2');
else    % write the data
    [M,N] = size(data);
    if 0
        for i=1:M
            if exponential
                fprintf(FID, '%e ', data(i,:));
            else
                fprintf(FID, '%f ', data(i,:));
            end
            %         for j=1:N
            %             fprintf(FID, '%d ',data(i,j));
            %         end
            fprintf(FID,'\n');
        end
    else
        if exponential
            formatStr = '%e ';
        else
            formatStr = '%f ';
        end
        formatStr = repmat(formatStr, 1, N);
        formatStr = [formatStr(1:end-1) '\n'];
        fprintf(FID, formatStr, data');
    end
end