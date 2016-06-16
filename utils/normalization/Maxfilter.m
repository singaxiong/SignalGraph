% Max filter

function y = Maxfilter(x,filter_len)

[nf,N_ch] = size(x);
hs = (filter_len-1)/2; % half side of the filter
y=x;
y(1:hs,:) = max(x(1:hs,:));
for i=hs+1:nf-hs
    y(i,:) = max(x(i-hs:i+hs,:));
end
y(end-hs+1:end,:) = max(x(end-hs+1:end,:));
