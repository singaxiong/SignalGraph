% Calculate the MFCC from filter bank
% Author: Xiao Xiong
% Created: 4 Feb 2005
% Last modified: 4 Feb 2005

function [feature] = fbank2mfcc(fbank,logE,DO_BLIND_EQUALIZATION);

global bias;
bias = zeros(12,1);

[N_vector, N_melBank]= size(fbank);

% for i=1:N_vector
%     % calculate the DCT of the vector. The mfcc sequences is
%     % c1,c2,c3...c12,c0
%     feature(i,1:13) = mydct(fbank(i,:), 13, N_melBank);
%     % do blind equalization to reduce convolutional distortion
%     if DO_BLIND_EQUALIZATION == 1
%         feature(i,1:12) = blind_equal(feature(i,1:12), logE(i));
%     end  
% end

% Faster implementation
feature(:,1:13) = mydct(fbank, 13, N_melBank);