% Median filter

function y = MovingAverageFilter(x,filter_len)

[nf,N_ch] = size(x);
hs = (filter_len-1)/2; % half side of the filter

x2 = [repmat(x(1,:), hs,1); x; repmat(x(end,:),hs,1)];
if 0
    y=x;
    for i=1:nf
        y(i,:) = mean(x2(i:i+2*hs,:));
    end
else
    overlap = filter_len-1;
    x_store = buffer(x2(overlap+1:end),filter_len,overlap);
    y = mean(x_store);
end

end
