% This function add a full matrix with a sparse matrix


function out = AddSpMatMat(w1,spMat, w2, Mat, sp_elements_only)
[m,n] = size(spMat);

if w1==-1
    spMat = -spMat;
elseif w1~=1
    spMat = spMat * w1;
end
if w2==-1
    Mat = -Mat;
elseif w2~=1
    Mat = Mat * w2;
end

out = Mat;
if 0
    idx = find(spMat);
    out(idx) = double(out(idx)) + spMat(idx);
elseif 1
    if m<n
        non_zero_col = find(sum(abs(spMat),1)>0);
        out(:,non_zero_col) = out(:,non_zero_col) + full(spMat(:,non_zero_col));
    else
        non_zero_row = find(sum(abs(spMat),2)>0);
        out(non_zero_row,:) = out(non_zero_row,:) + full(spMat(non_zero_row,:));
    end
else
    out = full(spMat) + Mat;
end
