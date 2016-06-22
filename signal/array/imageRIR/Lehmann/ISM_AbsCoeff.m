function [out,OKflag] = ISM_AbsCoeff(rttype,rt,room,weight,method,varargin)
%ISM_AbsCoeff  Calculates absorption coefficients for a given reverberation time
%
% [ALPHA,OKFLAG] = ISM_AbsCoeff(RT_TYPE,RT_VAL,ROOM,ABS_WEIGHT,METHOD) 
% [ALPHA,OKFLAG] = ISM_AbsCoeff( ... ,'c',SOUND_SPEED_VAL) 
%
% Returns the six absorption coefficients in the vector ALPHA for a given
% vector of room dimensions ROOM and a given value RT_VAL of reverberation
% time, with RT_TYPE corresponding to the desired measure of reverberation
% time, i.e., either 'T60' or 'T20'. Calling this function with RT_VAL=0
% simply returns ALPHA=[1 1 1 1 1 1] (anechoic case), regardless of the
% settings of the other input parameters.
%
% The parameter ABS_WEIGHTS is a 6 element vector of absorption
% coefficients weights which adjust the relative amplitude ratios between
% the six absorption coefficients in the resulting ALPHA vector. This
% allows the simulation of materials with different absorption levels on
% the room boundaries. Leave empty or set ABS_WEIGHTS=ones(1,6) to obtain
% uniform absorption coefficients for all room boundaries. 
%
% If the desired reverberation time could not be reached with the desired 
% environmental setup (i.e., practically impossible reverberation time 
% value given ROOM and ABS_WEIGHTS), the function will issue a warning on 
% screen accordingly. If the function is used with two output arguments, 
% the on-screen warnings are disabled and the function sets the flag OKFLAG 
% to 0 instead (OKFLAG is set to 1 if the computations are successful).
%
% The returned coefficients are calculated using one of the following
% methods, defined by the METHOD parameter:
%
%    * Lehmann and Johansson  (METHOD='LehmannJohansson')
%    * Sabine                 (METHOD='Sabine')
%    * Norris and Eyring      (METHOD='NorrisEyring')
%    * Millington-Sette       (METHOD='MillingtonSette')
%    * Fitzroy                (METHOD='Fitzroy')
%    * Arau                   (METHOD='Arau')
%    * Neubauer and Kostek    (METHOD='NeubauerKostek')
%
% In case the first computation method is selected (i.e., if METHOD is set
% to 'LehmannJohansson'), this function also accepts an additional
% (optional) argument 'c', which will set the value of the sound wave
% propagation speed to SOUND_SPEED_VAL. If omitted, 'c' will default to 343
% m/s. This parameter has no influence on the other six computation
% methods.
%
% Lehmann & Johansson's method relies on a numerical estimation of the
% energy decay in the considered environment, which leads to accurate RT
% prediction results. For more detail, see: "Prediction of energy decay in
% room impulse responses simulated with an image-source model", J. Acoust.
% Soc. Am., vol. 124(1), pp. 269-277, July 2008. The definition of T20 used
% with the 'LehmannJohansson' method corresponds to the time required by
% the energy--time curve to decay from -5 to -25dB, whereas the definition
% of T60 corresponds to the time required by the energy--time curve to
% decay by 60dB from the time lag of the direct path in the transfer
% function.
%
% On the other hand, the last six calculation methods are based on various
% established equations that attempt to predict the physical reverberation
% time T60 resulting from given environmental factors. These methods are
% known to provide relatively inaccurate results. If RT_TYPE='T20', the
% value of T20 for these methods then simply corresponds to T60/3 (linear
% energy decay assumption). For more information, see: "Measurement of
% Absorption Coefficients: Sabine and Random Incidence Absorption
% Coefficients" in the online room acoustics teaching material "AEOF3/AEOF4
% Acoustics of Enclosed Spaces" by Y.W. Lam, The University of Salford,
% 1995, as well as the paper: "Prediction of the Reverberation Time in
% Rectangular Rooms with Non-Uniformly Distributed Sound Absorption" by R.
% Neubauer and B. Kostek, Archives of Acoustics, vol. 26(3), pp. 183–202,
% 2001.

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

VarList = {'c'          343};	 % default sound propagation speed
eval(SetUserVars(VarList,varargin));

if ~strcmpi(rttype,'t60') && ~strcmpi(rttype,'t20'),
    error('Unrecognised ''RT_TYPE'' parameter (must be either ''T60'' or ''T20'').');
end

if rt==0,
    out = ones(size(weight));
    OKflag = 1;
    return
end

if isempty(weight), 
    weight = ones(1,6); 
else
    weight = weight./max(weight);
end

if strcmpi(method,'sabine'),
    if strcmpi(rttype,'t20'), rt = 3*rt; end        % linear energy decay assumption
    out = fminbnd(@sabine, 0.0001, 0.9999, [], rt, room, weight); 
elseif strcmpi(method,'norriseyring'),
    if strcmpi(rttype,'t20'), rt = 3*rt; end        % linear energy decay assumption
    out = fminbnd(@norris_eyring, 0.0001, 0.9999, [], rt, room, weight); 
elseif strcmpi(method,'millingtonsette'),
    if strcmpi(rttype,'t20'), rt = 3*rt; end        % linear energy decay assumption
    out = fminbnd(@millington_sette, 0.0001, 0.9999, [], rt, room, weight); 
elseif strcmpi(method,'fitzroy'),
    if strcmpi(rttype,'t20'), rt = 3*rt; end        % linear energy decay assumption
    out = fminbnd(@fitzroy, 0.0001, 0.9999, [], rt, room, weight); 
elseif strcmpi(method,'arau'),
    if strcmpi(rttype,'t20'), rt = 3*rt; end        % linear energy decay assumption
    out = fminbnd(@arau, 0.0001, 0.9999, [], rt, room, weight); 
elseif strcmpi(method,'neubauerkostek'),
    if strcmpi(rttype,'t20'), rt = 3*rt; end        % linear energy decay assumption
    out = fminbnd(@neubauer_kostek, 0.0001, 0.9999, [], rt, room, weight); 
elseif strcmpi(method,'lehmannjohansson'),
    if strcmpi(rttype,'t20'),
        out = fminbnd(@lehmann_johansson_20, 0.0001, 0.9999, [], rt, room, weight, c);
    else
        out = fminbnd(@lehmann_johansson_60, 0.0001, 0.9999, [], rt, room, weight, c);
    end
else
    error('Unrecognised ''METHOD'' parameter (see help for a list of accepted methods).');
end

foo = optimset('fminbnd'); 
tolx = foo.TolX;
if out<.0001+3*tolx,
    if nargout<2,
        warning(['Some absorption coefficients are close to the allowable limits (alpha->0). The \n' ...
                 'resulting reverberation time might end up lower than desired for the given environmental \n' ...
                 'setup. Try to relax some environmental constraints so that the desired reverberation time \n' ...
                 'is physically achievable (e.g., by increasing the room volume, increasing the maximum gap \n' ...
                 'between the absorption weights, or decreasing the desired RT value).%s'],'');
    end
    OKflag = 0;
elseif out>.9999-3*tolx,
    if nargout<2,
        warning(['Some absorption coefficients are close to the allowable limits (alpha->1). The \n' ...
                 'resulting reverberation time might end up higher than desired for the given environmental \n' ...
                 'setup. Try to relax some environmental constraints so that the desired reverberation time \n' ...
                 'is physically achievable (e.g., by reducing the room volume, reducing the maximum gap \n' ...
                 'between the absorption weights, or increasing the desired RT value).%s'],'');
    end
    OKflag = 0;
else
    OKflag = 1;
end
out = weight*out;


%======== Sabine's formula
function err=sabine(a,rt,room,weight)

alpha = a*weight;
V  = prod(room);       % Room volume
Sx = room(2)*room(3);  % Wall surface X
Sy = room(1)*room(3);  % Wall surface Y
Sz = room(1)*room(2);  % Wall surface Z
A  = Sx*(alpha(1) + alpha(2)) + Sy*(alpha(3) + alpha(4)) + Sz*(alpha(5) + alpha(6));
err = abs(rt - 0.161*V/A);  


%======== Millington-Sette's formula
function err=millington_sette(a,rt,room,weight)

alpha = a*weight;
V  = prod(room);       % Room volume
Sx = room(2)*room(3);  % Wall surface X
Sy = room(1)*room(3);  % Wall surface Y
Sz = room(1)*room(2);  % Wall surface Z
A  = -(Sx*(log(1-alpha(1)) + log(1-alpha(2))) + ...
       Sy*(log(1-alpha(3)) + log(1-alpha(4))) + ...
       Sz*(log(1-alpha(5)) + log(1-alpha(6))));
err = abs(rt - 0.161*V/A);


%======== Norris and Eyring's formula
function err=norris_eyring(a,rt,room,weight)

alpha = a*weight;
V  = prod(room);       % Room volume
Sx = room(2)*room(3);  % Wall surface X
Sy = room(1)*room(3);  % Wall surface Y
Sz = room(1)*room(2);  % Wall surface Z
St = 2*Sx+2*Sy+2*Sz;   % Total wall surface
A = Sx*(alpha(1) + alpha(2)) + Sy*(alpha(3) + alpha(4)) + Sz*(alpha(5) + alpha(6));
am = 1/St*A;
err = abs(rt + 0.161*V/(St*log(1-am)));


%======== Fitzroy's approximation
function err=fitzroy(a,rt,room,weight)

alpha = a*weight;
V  = prod(room);       % Room volume
Sx = room(2)*room(3);  % Wall surface X
Sy = room(1)*room(3);  % Wall surface Y
Sz = room(1)*room(2);  % Wall surface Z
St = 2*Sx+2*Sy+2*Sz;   % Total wall surface
tx = -2*Sx/log(1-mean(alpha(1:2)));
ty = -2*Sy/log(1-mean(alpha(3:4)));
tz = -2*Sz/log(1-mean(alpha(5:6)));
err = abs(rt - 0.161*V/(St^2)*(tx+ty+tz));


%======== Arau's formula
function err=arau(a,rt,room,weight)

alpha = a*weight;
V  = prod(room);       % Room volume
Sx = room(2)*room(3);  % Wall surface X
Sy = room(1)*room(3);  % Wall surface Y
Sz = room(1)*room(2);  % Wall surface Z
St = 2*Sx+2*Sy+2*Sz;   % Total wall surface
Tx = (0.161*V/(-St*log(1-mean(alpha(1:2)))))^(2*Sx/St);
Ty = (0.161*V/(-St*log(1-mean(alpha(3:4)))))^(2*Sy/St);
Tz = (0.161*V/(-St*log(1-mean(alpha(5:6)))))^(2*Sz/St);
err = abs(rt - (Tx*Ty*Tz));  


%======== Neubauer and Kostek's formula
function err=neubauer_kostek(a,rt,room,weight)

V  = prod(room);       % Room volume
Sx = room(2)*room(3);  % Wall surface X
Sy = room(1)*room(3);  % Wall surface Y
Sz = room(1)*room(2);  % Wall surface Z
St = 2*Sx+2*Sy+2*Sz;   % Total wall surface
r   = 1 - a*weight;
rww = mean(r(1:4));
rcf = mean(r(5:6));
rb  = mean(r);
aww = log(1/rb) + (r(1)*(r(1)-rww)*Sx^2 + r(2)*(r(2)-rww)*Sx^2 + ...
       r(3)*(r(3)-rww)*Sy^2 + r(4)*(r(4)-rww)*Sy^2)/((rww*(2*Sx+2*Sy))^2);
acf = log(1/rb) + (r(5)*(r(5)-rcf)*Sz^2 + r(6)*(r(6)-rcf)*Sz^2)/((rcf*2*Sz)^2);
err = abs(rt - 0.32*V/(St^2) * (room(3)*(room(1)+room(2))/aww + room(1)*room(2)/acf));


%======== Lehmann & Johannson's EDC approximation method
function err=lehmann_johansson_60(a,t60,room,weight,cc)

starttime = 1.4*mean(room)/cc;      % start time t0
DPtime = mean(room)/cc;             % direct path "estimate"
aa = a*weight;

numradpts = 60;
stoptime = 2*t60;
while 1,  % loop to determine appropriate stop time
    timepts = linspace(starttime,stoptime,numradpts);  % time points where to compute data

    [amppts1,timepts,okflag] = ISM_RIRpow_approx(aa,room,cc,timepts);
    
    for ii=1:length(amppts1),
        amppts1(ii) = sum(amppts1(ii:end));
    end
    amppts1 = 10*log10( amppts1/amppts1(1) );

    if amppts1(end)>=-60,
        if okflag==0,
            error('Problem computing EDC approximation!');
        end
        numradpts = numradpts + 30;     % more points are required for accurate T60 estimate
        stoptime = stoptime + t60;
        continue
    end

    sind = find(amppts1>=-60,1,'last');
    deltaX = timepts(2)-timepts(1);
    deltaY = amppts1(sind+1)-amppts1(sind);
    deltaA = -60 - amppts1(sind);
    t2 = timepts(sind) + deltaA*deltaX/deltaY;
    
    if t2>stoptime*2/3,
        numradpts = numradpts + 30;     % more points are required for accurate T60 estimate
        stoptime = stoptime + t60;
        if okflag==0,
            %error('Problem computing EDC approximation!');
            break   % use current time point if numerical limit is reached
        end
        continue
    else
        break
    end
end

t60est = t2-DPtime;
err = abs(t60 - t60est);


%======== Lehmann & Johannson's EDC approximation method
function err=lehmann_johansson_20(a,t20,room,weight,cc)

starttime = 1.4*mean(room)/cc;        % start time t0
aa = a*weight;

numradpts = 40;
stoptime = 5*t20;
while 1,  % loop to determine appropriate stop time
    timepts = linspace(starttime,stoptime,numradpts);  % time points where to compute data

    [amppts1,timepts,okflag] = ISM_RIRpow_approx(aa,room,cc,timepts);
    
    for ii=1:length(amppts1),
        amppts1(ii) = sum(amppts1(ii:end));
    end
    amppts1 = 10*log10( amppts1/amppts1(1) );

    if amppts1(end)>=-25,
        if okflag==0,
            error('Problem computing EDC approximation!');
        end
        numradpts = numradpts + 30;     % more points are required for accurate T20 estimate
        stoptime = stoptime + 3*t20;
        continue
    end

    sind = find(amppts1>=-5,1,'last');
    deltaX = timepts(2)-timepts(1);
    deltaY = amppts1(sind+1)-amppts1(sind);
    deltaA = -5 - amppts1(sind);
    t1 = timepts(sind) + deltaA*deltaX/deltaY;

    sind = find(amppts1>=-25,1,'last');
    deltaY = amppts1(sind+1)-amppts1(sind);
    deltaA = -25 - amppts1(sind);
    t2 = timepts(sind) + deltaA*deltaX/deltaY;
    
    if t2>stoptime*2/3,
        numradpts = numradpts + 30;     % more points are required for accurate T20 estimate
        stoptime = stoptime + 3*t20;
        if okflag==0,
            %error('Problem computing EDC approximation!');
            break   % use current time point if numerical limit is reached
        end
        continue
    else
        break
    end
end

t20est = t2-t1;
err = abs(t20 - t20est);
