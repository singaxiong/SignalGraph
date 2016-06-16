% This function add a full matrix with a sparse matrix


function out = AddSpMatMat_sparseonly(w1,spMat, w2, Mat)
[m,n] = size(spMat);

if w1==-1
    spMat = -spMat;
elseif w1~=1
    spMat = spMat * w1;
end

if 0
    if w2==-1
        Mat = -Mat;
    elseif w2~=1
        Mat = Mat * w2;
    end
    idx = find(spMat);    % find the nonzero elemetns in linear indexing
    [idx1, idx2] = find(spMat);   % find the nonzero elements in 2D indexing
    Mat2 = double(Mat(idx));
    Mat_sparse= sparse(idx1,idx2,Mat2,m,n);
    out = spMat + Mat_sparse;
else
    out = sparse(m,n);
    if m<n
        non_zero_col = find(sum(abs(spMat),1)>0);
        out(:,non_zero_col) = w2*Mat(:,non_zero_col) + full(spMat(:,non_zero_col));
    else
        non_zero_row = find(sum(abs(spMat),2)>0);
        out(:,non_zero_col) = w2*Mat(non_zero_row,:) + full(spMat(non_zero_row,:));
    end
end

end


% function out = AddSpMatMatCore(spMat, Mat, sp_elements_only)
% [m,n] = size(spMat);
% if m<n
%     non_zero_col = find(sum(abs(spMat),1)>0);
%     out(:,non_zero_col) = out(:,non_zero_col) + full(spMat(:,non_zero_col));
% else
%     non_zero_row = find(sum(abs(spMat),2)>0);
%     out(non_zero_row,:) = out(non_zero_row,:) + full(spMat(non_zero_row,:));
% end

% end

