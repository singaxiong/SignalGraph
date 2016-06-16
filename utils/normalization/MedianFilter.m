% Median filter

function y = MedianFilter(x,filter_len)

[nf,N_ch] = size(x);
hs = (filter_len-1)/2; % half side of the filter
y=x;
for i=hs+1:nf-hs
    y(i,:) = median(x(i-hs:i+hs,:));
end