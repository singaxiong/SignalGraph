function xfinal = my_ola(xi, len, len1)
len2 = len - len1;

nFr = size(xi,2);
k=1;
xfinal=zeros(nFr*len2,1);
x_old = zeros(len1,1);
for j=1:nFr
%     plot(x_old); hold on;
%     plot(xi(1:len1),'r'); hold off; 
%     legend('x_old', 'xi'); pause;
    xfinal(k:k+len1-1) = x_old + xi(1:len1,j);
%     xfinal(k:k+len1-1) = xi(1:len1,j);
    x_old = xi(1+len2:len,j);
    k = k + len2;
    
%     plot(xfinal(1:k)); pause;
end
