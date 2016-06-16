% Compute the DCT of the filter bank. The program implements the DCT
% computation of the WI007 of AURORA project. 
% There will usually be 13 DCT coefficients. The first coefficient is
% computed and appended after the other two coefficients. so the sequence
% of the coefficients is c1c2...c12c0
% Author: Xiao Xiong
% Created: 6 Feb 2005
% Last modified: 6 2005

function [out] = mydct(data, num_MFCC, num_Fbank);

% STEP 1: create the transform matrix for c1-c12 first
% for i = 1:num_MFCC-1
%    for j=1:num_Fbank
%        Mx(i,j) = cos ( pi*i/num_Fbank * (j-0.5) );
%    end
% end

% Faster implementation: don't use for loop
i = (1:num_MFCC-1)';
j=1:num_Fbank;
Mx = cos ( pi*i/num_Fbank * (j-0.5) );

% STEP 2: compute the dct coefficients
% for i = 1:num_MFCC-1
%     out(i) = Mx(i,:)*data';
% end
% Faster implementatoin: vectorize
out = Mx*data';

% comppute the c0
out(num_MFCC,:) = sum(data,2);
out = out';