function w = my_hamming(varargin)
%HAMMING   Hamming window.
%   HAMMING(N) returns the N-point symmetric Hamming window in a column vector.
% 
%   HAMMING(N,SFLAG) generates the N-point Hamming window using SFLAG window
%   sampling. SFLAG may be either 'symmetric' or 'periodic'. By default, a 
%   symmetric window is returned. 
%
%   See also BLACKMAN, HANN, WINDOW.

%   Copyright 1988-2002 The MathWorks, Inc.
%   $Revision: 1.14.4.3 $  $Date: 2011/05/13 18:07:55 $

% Check number of inputs
narginchk(1,2);

[w,msg,msgobj] = my_gencoswin('hamming',varargin{:});
if ~isempty(msg), error(msgobj); end


% [EOF] hamming.m
