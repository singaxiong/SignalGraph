function [y invMx] = myidct(x, num_MFCC, num_Fbank)

i = (1:num_Fbank-1)';
j=1:num_Fbank;
Mx = cos ( pi*i/num_Fbank * (j-0.5) );
Mx  = [Mx(1:num_MFCC-1,:); ones(1, num_Fbank); Mx(num_MFCC:end,:)];
invMx = inv(Mx);
invMx = invMx(:,1:num_MFCC);
y = invMx * x';
y = y';