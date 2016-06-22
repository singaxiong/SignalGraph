function [amppts1,timepts,okflag] = ISM_RIRpow_approx(aa,room,cc,timepts,rt_type,rt_val)
%ISM_RIRpow_approx  Approximation of ISM RIR power (Lehmann & Johansson's method)
%
% [P_VEC,T_VEC,OK_FLAG] = ISM_RIRpow_approx(ALPHA,ROOM,C,T_VEC,RT_TYPE,RT_VAL)
% 
% This function returns the predicted values of RIR power in P_VEC (as
% would result from ISM simulations) estimated by means of the EDC
% approximation method described in: "Prediction of energy decay in room
% impulse responses simulated with an image-source model", J. Acoust. Soc.
% Am., vol. 124(1), pp. 269-277, July 2008. The values of P_VEC are
% computed for the time points given as input in T_VEC (in sec), which is
% assumed to contain increasing values of time. The vector T_VEC (and
% corresponding vector P_VEC) will be cropped if the numerical computation
% limits are reached for the higher time values in T_VEC (for which NaNs
% are generated in P_VEC), in which case the output parameter OK_FLAG will
% be set to 0 (1 otherwise).
%
% The environmental setting is defined via the following input parameters:
%
%    ALPHA: 1-by-6 vector, corresponding to each wall's absorption 
%           coefficient: [x1 x2 y1 y2 z1 z2]. Index 1 indicates wall closest
%           to the origin. E.g.: [0.5 0.5 0.45 0.87 0.84 0.32].
%  RT_TYPE: character string, measure of reverberation time used for the 
%           definition of the coefficients in ALPHA. Set to either 'T60' or
%           'T20'. 
%   RT_VAL: scalar, value of the reverberation time (in seconds) defined by
%           RT_TYPE. E.g.: 0.25.
%     ROOM: 1-by-3 vector, indicating the rectangular room dimensions 
%           (in m): [x_length y_length z_length]. E.g.: [4 4 3].
%        C: scalar (in m/s), propagation speed of sound waves. E.g.: 343. 

% Release date: November 2009
% Author: Eric A. Lehmann, Perth, Australia (www.eric-lehmann.com)
%
% Copyright (C) 2009 Eric A. Lehmann
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
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

numradpts = length(timepts);
radpts = cc * timepts;              % radius values corresponding to time points

bxx = ( sqrt(1-aa(1))*sqrt(1-aa(2)) )^(1/room(1));
byy = ( sqrt(1-aa(3))*sqrt(1-aa(4)) )^(1/room(2));
bzz = ( sqrt(1-aa(5))*sqrt(1-aa(6)) )^(1/room(3));

if bxx==byy && byy==bzz,
    intcase = 1;
elseif bxx==byy && bxx~=bzz,
    intcase = 2;
elseif byy==bzz && bzz~=bxx,
    if bzz<bxx,     % coordinate swap x<->z
        foo = bxx; bxx = bzz; bzz = foo;
        intcase = 2;
    else
        intcase = 3;
    end
elseif bxx==bzz && bzz~=byy,
    if bzz<byy,     % coordinate swap y<->z
        foo = byy; byy = bzz; bzz = foo;
        intcase = 2;
    else
        intcase = 4;
    end
else
    intcase = 5;
    if bxx>bzz && bxx>byy,      % coordinate swap x<->z
        foo = bxx; bxx = bzz; bzz = foo;
    elseif byy>bzz && byy>bxx,	% coordinate swap y<->z
        foo = byy; byy = bzz; bzz = foo;
    end
end

amppts1 = zeros(1,numradpts);
for ss=1:numradpts,    % compute amplitude/energy estimates
    Bx = bxx^(radpts(ss)); Bx(Bx==0) = eps;
    By = byy^(radpts(ss)); By(By==0) = eps;
    Bz = bzz^(radpts(ss)); Bz(Bz==0) = eps;
    switch intcase
        case 1
            int2 = Bx;
        case 2
            int2 = (Bx-Bz) / log(Bx/Bz);
        case 3
            n1 = log(Bz/Bx);
            int2 = Bz*( expint(n1) + log(n1) + 0.5772156649 ) / n1;
        case 4
            n1 = log(Bz/By);
            int2 = Bz*( expint(n1) + log(n1) + 0.5772156649 ) / n1;
        otherwise
            n1 = log(Bz/By);
            n2 = log(Bz/Bx);
            int2 = Bz*(log(n1/n2) + expint(n1) - expint(n2)) / log(Bx/By);
    end
    amppts1(ss) = int2/radpts(ss);      % 'propto' really...
end

okflag = 1;
foo = find(isnan(amppts1),1);
if ~isempty(foo),
    amppts1 = amppts1(1:foo-1);
    timepts = timepts(1:foo-1);
    okflag = 0;
end

if nargin==6,
    switch lower(rt_type)  % offset correction
        case 't60', sl = exp(3.05*exp(-1.85*rt_val));
        case 't20', sl = exp(3.52*exp(-7.49*rt_val));
    end
    amppts1 = amppts1 ./ exp(sl*(timepts-timepts(1)));
end
