function [RIRt,okf] = fast_ISM_RoomResp(OutStruct,ch)
%fast_ISM_RoomResp  Fast computation of image-source RIRs using Lehmann & Johansson's method
%
% [RIR,OK_FLAG] = fast_ISM_RoomResp(Fs,BETA,RT_TYPE,RT_VAL,SOURCE,SENSOR,ROOM)
% [RIR,OK_FLAG] = fast_ISM_RoomResp( ... ,'arg1',val1,'arg2',val2,...)
% 
% This function generates the room impulse response (RIR) between a sound
% source and an acoustic sensor, based on various environmental parameters
% such as source and sensor positions, enclosure's dimensions and
% reflection coefficients, etc. The simulation method is based on a model
% of the RIR where the early reflections are computed according to Lehmann
% and Johansson's implementation of the image-source method (see
% "Prediction of energy decay in room impulse responses simulated with an
% image-source model", J. Acoust. Soc. Am., vol. 124(1), pp. 269-277, July
% 2008), and the late reflections (diffuse field) are generated as decaying
% random noise where the energy decay is determined in accordance with
% full-ISM simulated results (see "Diffuse reverberation model for efficient 
% image-source simulation of room impulse responses", IEEE Trans. Audio, 
% Speech, Lang. Process., 2010). 
%
% The input parameters are defined as follows:
%
%       Fs: scalar, sampling frequency (in Hz). E.g.: 8000.
%     BETA: 1-by-6 vector, corresponding to each wall's reflection 
%           coefficient: [x1 x2 y1 y2 z1 z2]. Index 1 indicates wall closest
%           to the origin. This function assumes strictly non-negative BETA
%           coefficients. Set to [0 0 0 0 0 0] to obtain anechoic response 
%           (direct path only), in which case the value of RT_VAL is 
%           discarded. E.g.: [0.75 0.5 0.6 0.7 0.65 0.6].
%  RT_TYPE: character string, measure of reverberation time used for the 
%           definition of the coefficients in BETA. Set to either 'T60' or
%           'T20'. 
%   RT_VAL: scalar, value of the reverberation time (in seconds) defined by
%           RT_TYPE. Set to 0 to obtain anechoic response (same effect as 
%           setting BETA to [0 0 0 0 0 0]), in which case the BETA 
%           coefficients are discarded. E.g.: 0.25.
%   SOURCE: 1-by-3 vector, indicating the location of the source in space 
%           (in m): [x y z]. E.g.: [1 1 1.5].
%   SENSOR: 1-by-3 vector, indicating the location of the microphone in 
%           space (in m): [x y z]. E.g.: [2 2 1.5].
%     ROOM: 1-by-3 vector, indicating the rectangular room dimensions 
%           (in m): [x_length y_length z_length]. E.g.: [4 4 3].
%
% In addition, a number of other (optional) parameters can be set using a 
% series of 'argument'--value pairs. The following parameters (arguments)
% can be used:
%
%          'c': scalar, speed of acoustic waves (in m/s). Defaults to 343.
%   'Delta_dB': scalar (in dB), parameter determining how much the 
%               resulting impulse response is cropped: the impulse response
%               is computed until the time index where its overall energy
%               content has decreased by 'Delta_dB' decibels, after which 
%               the computations stop. Not relevant if BETA=zeros(1,6).
%               Defaults to 50.
% 'Diffuse_dB': scalar (in dB), determines the desired transition point from
%               early reflections to diffuse field: the diffuse reflections
%               are assumed to start from the time index where the RIR's
%               energy has decayed by 'Diffuse_dB' decibels, following
%               which the diffuse field model is applied (see above
%               mentioned reference). Not relevant if BETA=zeros(1,6).
%               Defaults to 13.
% 
% This function returns the time coefficients of the filter (transfer
% function) in the parameter RIR. The filter coefficients are real and
% non-normalised. The first value in the vector RIR, i.e., RIR(1),
% corresponds to time t=0. The number of coefficients returned is variable
% and results from the value of 'Delta_dB' defined by the user: the filter
% length will be as large as necessary to capture all the relevant
% highest-order reflections. 
%
% The simulation approach implemented by this function produces RIRs with a
% proper energy decay and statistical tail distribution, while reducing the
% simulation times by up to two orders of magnitude compared to a
% full-length ISM computation. Note however that the diffuse field model
% used here is only an approximation based on the predicted _average_ tail
% decay. Therefore, some particular RIRs, e.g., those exhibiting an
% irregular energy decay pattern, cannot be properly modeled with the above
% technique. This is typically the case in environments exhibiting a very
% uneven distribution of the sound absorption among the walls. In such
% cases, a full-ISM simulation is therefore necessary in order to obtain
% accurate results (use the Matlab function 'ISM_RoomResp.m' instead). See
% the above mentioned references for more information. This function checks
% the environmental parameters given as input to detect whether such an
% irregular decay pattern is likely to occur, and in such a case, it will
% print a warning message on screen and set OK_FLAG to 0 (1 otherwise).

% Release date: March 2012
% Author: Eric A. Lehmann, Perth, Australia (www.eric-lehmann.com)
%
% Copyright (C) 2012 Eric A. Lehmann
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

Fs = OutStruct.samplingFreq;
rt_type = 'T60';
rt_val = OutStruct.t60;
beta = OutStruct.reflectCoeffs;
X_src = OutStruct.srcPosition;
X_rcv = OutStruct.micPosition(ch,:);
t60 = OutStruct.t60;
room = OutStruct.roomDimension;
c = OutStruct.soundVelocity;
Delta_dB = 60;
Diffuse_dB=13; 
% ,
SilentFlag=1;
DisableIrregDecayWarnings=1;

% VarList = {'c'                  343;	 % sound propagation speed
%            'Diffuse_dB'         13;      % diffuse field limit
%            'Delta_dB'           50; 	 % attenuation limit
%            'DisableIrregDecayWarnings'  0};	% [undocumented -- use at your own risk!] disable warnings related to irregular 
%                                             % RIR decay patterns (which cannot be modeled properly with the current function!).
% eval(SetUserVars(VarList,varargin));

%-=:=- Check user input -=:=-
if X_rcv(1)>=room(1) || X_rcv(2)>=room(2) || X_rcv(3)>=room(3) || X_rcv(1)<=0 || X_rcv(2)<=0 || X_rcv(3)<=0,
    error('Receiver must be within the room boundaries!');
elseif X_src(1)>=room(1) || X_src(2)>=room(2) || X_src(3)>=room(3) || X_src(1)<=0 || X_src(2)<=0 || X_src(3)<=0,
    error('Source must be within the room boundaries!');
elseif ~isempty(find(beta>=1,1)) || ~isempty(find(beta<0,1)),
    error('Parameter ''BETA'' must be in the range [0...1).');
end
if ~strcmpi(rt_type,'t60') && ~strcmpi(rt_type,'t20'),
    error('Unknown RT_TYPE parameter.');
end

%-=:=- Anechoic response only -=:=-
if isequal(beta(:),zeros(6,1)) || rt_val==0,      % anechoic case: compute direct path only
	RIRt = ISM_RoomResp(Fs,beta,rt_type,rt_val,X_src,X_rcv,room,'SilentFlag',1,'c',c);
	okf = 1;
	return
end

%-=:=- Check for irregular decay pattern -=:=-
alpha = 1 - beta.^2;
Sx = room(2)*room(3); Sy = room(1)*room(3); Sz = room(1)*room(2);
Avec = [Sx*alpha(1) Sx*alpha(2) Sy*alpha(3) Sy*alpha(4) Sz*alpha(5) Sz*alpha(6)];
irreg_crit = std(Avec/(2*(Sx+Sy+Sz)));
okf = 1;
if irreg_crit>0.035,%%0.04,
    if ~DisableIrregDecayWarnings,
        warning(['the uneven distribution of sound absorption in the considered environment might \n' ...
            'lead to an irregular decay pattern in the computed RIR, which cannot be modeled properly \n' ...
            'with the current approximation method. Use of a full ISM simulation (see ''ISM_RoomResp.m'') \n' ...
            'is recommended in order to achieve a proper impulse response.%s'],'');
    end
    okf = 0;
end

%=:=:=:=:==:=:=:=:=:=:= Time limits =:=:=:=:==:=:=:=:=:=
foo = ISM_RIR_DecayTime([Diffuse_dB Delta_dB],rt_type,rt_val,alpha,room,X_src,X_rcv,Fs,c);
diff_DT = foo(1);
RIRt_stoptime = foo(2);
RIRt_stoptime_samp = ceil(RIRt_stoptime*Fs);
dp_del = norm(X_src-X_rcv)/c;   % direct path delay
dp_del_samp = round(dp_del*Fs);
starttime = 1.4*mean(room)/c;   % time t0 of first energy window
starttime_sind = round(starttime*Fs);

%=:=:=:=:==:=:=:=:= Early reflections =:=:=:=:==:=:=:=:=
RIRe = ISM_RoomResp(Fs,beta,rt_type,rt_val,X_src,X_rcv,room,'SilentFlag',1,'c',c,'MaxDelay',diff_DT);
RIRe_len = length(RIRe);

%=:=:=:=:= Diffuse field (late reverberations) =:=:=:=:=
%-=:=- Select window size -=:=-
num_win = max([5 Diffuse_dB*0.8]);  % nr of windows determined roughly by Diffuse_dB
TT = (diff_DT-dp_del)/num_win;      % some appropriate window length
w_len = ceil(TT*Fs);                % current window length (samples)
w_len = w_len + mod(w_len,2) - 1;	% make w_len odd
w_len_half = floor(w_len/2);

%-=:=- Measure average energy -=:=-
if (starttime_sind+w_len_half)>RIRe_len,
    error('Problem when computing diffuse reverberations (parameter ''Diffuse_dB'' might be too small).');
end
we_sind_vec = [starttime_sind+w_len_half:w_len:RIRe_len];  % window end indices
wc_sind_vec = we_sind_vec - w_len_half;                 % window centre indices
wb_sind_vec = wc_sind_vec - w_len_half;                 % window beginning indices
if wb_sind_vec(1)<=0, wb_sind_vec(1) = 1; end           % case where t0 is less than a half window
n_win_meas = length(wc_sind_vec);

en_vec_meas = zeros(1,n_win_meas);
for ww=1:n_win_meas,
    en_vec_meas(ww) = mean( RIRe(wb_sind_vec(ww):we_sind_vec(ww)).^2 );
end

%-=:=- Compute approximated energy function -=:=-
fit_time_perc = .3;    % fit start time as %-age of RIR between dp-del and RIRe_len
fit_start_sind = fit_time_perc*(RIRe_len-dp_del_samp) + dp_del_samp;
fit_start_wind = find(wc_sind_vec>=fit_start_sind,1);   % window index of start of fit
wc_sind_vec_approx = [starttime_sind:w_len:RIRt_stoptime_samp+w_len-1];
t_vec_approx = wc_sind_vec_approx/Fs;

[en_vec_approx,t_vec_approx] = ISM_RIRpow_approx(alpha,room,c,t_vec_approx,rt_type,rt_val);
wc_sind_vec_approx = round(t_vec_approx*Fs);   % simply use available time points for reconstruction...

%-=:=- Lambda-adjust approximated energy function -=:=-
foo = en_vec_meas(fit_start_wind:end) ./ en_vec_approx(fit_start_wind:n_win_meas);
en_vec_approx = en_vec_approx * mean(foo);

%-=:=- Reconstruct RIR (abrupt change) -=:=-
rec_time_perc = .9;    % rec start time as %-age of RIR between dp-del and RIRe_len
rec_start_sind = rec_time_perc*(RIRe_len-dp_del_samp) + dp_del_samp;
rec_start_wind = find(wc_sind_vec>=rec_start_sind,1);   % window index of start of reconstruction
if isempty(rec_start_wind),         % make rec_start_sind the center of a window
    rec_start_sind = wc_sind_vec(end);
else
    rec_start_sind = wc_sind_vec(rec_start_wind);
end
RIRd_sind_vec = [rec_start_sind+1:RIRt_stoptime_samp];	% sample indices of reconstructed RIR (diffuse)

en_vec_RIRd = interp1(wc_sind_vec_approx,en_vec_approx,RIRd_sind_vec);
RIRd = sqrt(en_vec_RIRd).*logistic_rnd(1,RIRt_stoptime_samp-rec_start_sind,0,sqrt(3)/pi);
RIRt = [RIRe(1:rec_start_sind);  RIRd.'];
