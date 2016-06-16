function y = mat2vec(x)
[m,n] = size(x);
y = reshape(x,m*n,1);