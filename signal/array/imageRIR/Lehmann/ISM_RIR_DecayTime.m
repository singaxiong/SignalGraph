function DTime_vec = ISM_RIR_DecayTime(delta_dB_vec,rt_type,rt_val,aa,room,X_src,X_rcv,Fs,cc)
%ISM_RIR_DecayTime  RIR decay time using Lehmann & Johansson's EDC approximation method
%
% DT = ISM_RIR_DecayTime(DELTA_dB,RT_TYPE,RT_VAL,ALPHA,ROOM,SOURCE,SENSOR,Fs)
% DT = ISM_RIR_DecayTime(DELTA_dB,RT_TYPE,RT_VAL,ALPHA,ROOM,SOURCE,SENSOR,Fs,C)
%
% This function determines the time DT taken by the energy in a RIR to
% decay by DELTA_dB when using Lehmann & Johansson's image-source method
% implementation (see: "Prediction of energy decay in room impulse
% responses simulated with an image-source model", J. Acoust. Soc. Am.,
% vol. 124(1), pp. 269-277, July 2008). The parameter DELTA_dB can be
% defined as a vector of dB values, in which case this function returns DT
% as a vector containing the corresponding decay times. Note that DT does
% not necessarily correspond to the usual definition of quantities such as
% T20, T30, or T60. The resulting DT values are computed according to
% Lehmann and Johannson's EDC (energy decay curve) approximation method
% (see above reference) used in conjunction with a RIR reconstruction
% method based on diffuse reverberation modeling (see "Diffuse 
% reverberation model for efficient image-source simulation of room 
% impulse responses", IEEE Trans. Audio, Speech, Lang. Process., 2010).
%
% The environmental room setting is given via the following parameters:
%
%       Fs: scalar, sampling frequency (in Hz). Eg: 8000.
%    ALPHA: 1-by-6 vector, corresponding to each wall's absorption 
%           coefficient: [x1 x2 y1 y2 z1 z2]. Index 1 indicates wall 
%           closest to the origin. E.g.: [0.5 0.5 0.45 0.87 0.84 0.32].
%  RT_TYPE: character string, measure of reverberation time used for the 
%           definition of the coefficients in ALPHA. Set to either 'T60' or
%           'T20'. 
%   RT_VAL: scalar, value of the reverberation time (in seconds) defined by
%           RT_TYPE. E.g.: 0.25.
%   SOURCE: 1-by-3 vector, indicating the location of the source in space 
%           (in m): [x y z]. E.g.: [1 1 1.5].
%   SENSOR: 1-by-3 vector, indicating the location of the microphone in 
%           space (in m): [x y z]. E.g.: [2 2 1.5].
%     ROOM: 1-by-3 vector, indicating the rectangular room dimensions 
%           (in m): [x_length y_length z_length]. E.g.: [4 4 3].
%        C: (optional) scalar (in m/s), propagation speed of sound waves.   
%           If omitted, C will default to 343m/s. 

% Release date: March 2012
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

if nargin==8,
    cc = 343;
end
delta_dB_vec = abs(delta_dB_vec);

%-=:=- Check user input:
if ~isempty(find(aa>1,1)) || ~isempty(find(aa<=0,1)),
    error('Parameter ''ALPHA'' must be in range (0...1].');
elseif ~isempty(find(delta_dB_vec==0,1)),
    error('Parameter ''DELTA_dB'' must contain non-zero scalar values.');
elseif min(size(delta_dB_vec))~=1,
    error('Parameter ''DELTA_dB'' must be a 1-D vector.');
end

if isequal(aa(:),ones(6,1)) || rt_val==0,
	error('ISM_RIR_DecayTime cannot be used for anechoic environments.');
end

switch lower(rt_type)
    case 't60'
        t60_appval = rt_val;
    case 't20'
        t60_appval = rt_val*3;  % coarse t60 estimate to determnine end time in EDCapprox computations
    otherwise
        error('Unknown ''RT_TYPE'' argument.');
end

%-=:=- Pre-processing -=:=-
dp_del = norm(X_src-X_rcv)/cc;          % direct path
delta_dB_max = max(delta_dB_vec);
n_ddbv = length(delta_dB_vec);
starttime = max([1.4*mean(room)/cc dp_del]);    % start time t0, ensure >= dp_delay
starttime_sind = floor(starttime*Fs);   % sample index
RIR_start_DTime = 2*starttime;

%-=:=- select window size -=:=-
n_win_meas = 6;                     % approximate nr of (useful) measurements
TT = (RIR_start_DTime-starttime)/n_win_meas;
w_len = floor(TT*Fs);                % current window length (samples)
w_len = w_len + mod(w_len,2) - 1;	% make w_len odd
w_len_half = floor(w_len/2);

%-=:=- pre-compute start of RIR for lambda correction -=:=-
RIR = ISM_RoomResp(Fs,sqrt(1-aa),rt_type,rt_val,X_src,X_rcv,room,'SilentFlag',1,'MaxDelay',RIR_start_DTime,'c',cc);
RIRlen = length(RIR);

%-=:=- Measure average energy -=:=-
fit_time_perc = 0.35;
we_sind_vec = [starttime_sind+w_len_half:w_len:RIRlen]; % window end indices
wc_sind_vec = we_sind_vec - w_len_half;                 % window centre indices
wb_sind_vec = wc_sind_vec - w_len_half;                 % window beginning indices
if wb_sind_vec(1)<=0, wb_sind_vec(1) = 1; end           % case where t0 is less than a half window
n_win_meas = length(wc_sind_vec);
en_vec_meas = zeros(1,n_win_meas);
for ww=1:n_win_meas,
    en_vec_meas(ww) = mean( RIR(wb_sind_vec(ww):we_sind_vec(ww)).^2 );
end
t_vec_meas = wc_sind_vec/Fs;
fit_starttime = RIRlen*fit_time_perc/Fs;
fit_start_wind = find(t_vec_meas>=fit_starttime,1);   % window index of start of fit

%-=:=- Decay time estimate -=:=-
DTime_vec = NaN*ones(size(delta_dB_vec));
stind = 3; 
while stind>0,
    % compute + lambda-adjust EDC approximation
    stoptime = stind * delta_dB_max/60*t60_appval;     % IMapprox computed up to several times what linear decay predicts
    timepts = [starttime_sind:w_len:stoptime*Fs]/Fs;
    stoptime = timepts(end);
    [amppts1,timepts,okflag] = ISM_RIRpow_approx(aa,room,cc,timepts,rt_type,rt_val);      % compute EDC approximation
    foo = en_vec_meas(fit_start_wind:end) ./ amppts1(fit_start_wind:n_win_meas);      % offset compensation (lambda)
    amppts1 = amppts1 * mean(foo);

    % reconstruct approx. full RIR for proper EDC estimation (logistic noise approx.)
    amppts1_rec = interp1(timepts,amppts1,[RIRlen+1:stoptime*Fs]/Fs);
    RIR_rec = [RIR.' sqrt(amppts1_rec)];
    RIR_rec_len = length(RIR_rec);
    
    % approx. full RIR EDC
    edc_rec = zeros(1,RIR_rec_len);
    for nn=1:RIR_rec_len,
        edc_rec(nn) = sum(RIR_rec(nn:end).^2);  % Energy decay using Schroeder's integration method
    end
    edc_rec = 10*log10(edc_rec/edc_rec(1));     % Decay curve in dB.
    tvec_rec = [0:RIR_rec_len-1]/Fs;

    % Determine time of EDC reaching delta_dB decay:
    if edc_rec(end)>-delta_dB_max,
        stind = stind + 1;
        if okflag==0, error('Problem computing decay time (parameter ''DELTA_dB'' may be too large)'); end
        if stind>=25, error('Problem computing decay time (parameter ''DELTA_dB'' may be too large)'); end
        continue
    end

    for nn=1:n_ddbv,
        foo = find(edc_rec<=-delta_dB_vec(nn),1);
        DTime_vec(nn) = 1.15*tvec_rec(foo);     % statistical offset correction...
    end
    
    if max(DTime_vec)>stoptime*2/3,    % make sure IM approx was computed for more than 3/2 the resulting decay time.
        stind = stind + 1;    % increase time if necessary
        if okflag==0, error('Problem computing decay time (parameter ''DELTA_dB'' may be too large)'); end
        if stind>=25, error('Problem computing decay time (parameter ''DELTA_dB'' may be too large)'); end
        continue
    else
        stind = 0;          % stop the computations
    end
end
