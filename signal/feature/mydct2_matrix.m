% Compute the DCT of the filter bank. The program implements the DCT
% computation of the WI007 of AURORA project. 
% There will usually be 13 DCT coefficients. The first coefficient is
% computed and appended after the other two coefficients. so the sequence
% of the coefficients is c1c2...c12c0
% Author: Xiao Xiong
% Created: 6 Feb 2005
% Last modified: 6 2005

function Mx = mydct2_matrix(M, N)

m=0:M-1;
n=0:N-1;
Mx = zeros(M*N,M*N);

cnt = 1;
for p = 0:M-1
    for q = 0:N-1
        if p == 0
            ap = sqrt(1/M);
        else
            ap = sqrt(2/M);
        end
        if q == 0
            aq = sqrt(1/N);
        else
            aq = sqrt(2/N);
        end

        Bpq = ap*aq* (cos(pi*(2*m+1)*p/2/M)') * (cos(pi*(2*n+1)*q/2/N)')';
        Mx(:,cnt) = reshape(Bpq, M*N,1);
        cnt = cnt + 1;
    end
end