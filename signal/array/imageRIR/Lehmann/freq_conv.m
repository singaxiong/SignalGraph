function rr = freq_conv(xx,yy)
%freq_conv  Performs convolution in frequency domain
%
% C = freq_conv(X,Y)
%
% This function computes the convolution of X and Y in the frequency domain
% and returns the result in C. Both X and Y must be one-dimensional (line
% or column) vectors. The format of the output C (i.e., line or column
% vector) matches that of the vector X.
%
% This function allows significant savings in execution time compared to
% the time-domain equivalent, i.e., Matlab's conv function.

% Release date: August 2008
% Author: Eric A. Lehmann, Perth, Australia (www.eric-lehmann.com)
%
% Copyright (C) 2008 Eric A. Lehmann
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.

xxs = size(xx); yys = size(yy);
if min(xxs)~=1 || min(yys)~=1,
    error('Both input vectors must be one-dimensional.');
end

xx = xx(:); yy = yy(:);
rlen = length(xx)+length(yy)-1;
rlen_p2 = 2^nextpow2(rlen);
XX = fft(xx,rlen_p2);
YY = fft(yy,rlen_p2);
rr = ifft(XX.*YY,'symmetric');
rr = rr(1:rlen);    %column vector

if xxs(1)==1,   % output rr in same format as xx
    rr = rr.';
end
