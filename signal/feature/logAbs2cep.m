% Calculate the MFCC from filter bank
% Author: Xiao Xiong
% Created: 4 Feb 2005
% Last modified: 4 Feb 2005

function ceps = logAbs2cep(logAbs,N_ceps);

[N_vec, N_freq_Bank]= size(logAbs);

for i=1:N_vec
    % calculate the DCT of the vector. The mfcc sequences is
    % c1,c2,c3...c12,c0
    ceps(i,1:N_ceps) = mydct(logAbs(i,:), N_ceps, N_freq_Bank);
%     a = mydct(logAbs(i,:), 13, N_freq_Bank);
%     b = dct(logAbs(i,:), 13);
end
