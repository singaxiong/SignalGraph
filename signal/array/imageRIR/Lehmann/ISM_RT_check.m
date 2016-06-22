function [medRT,RTvec] = ISM_RT_check(setupstruc,varargin)
%ISM_RT_check  Reverberation time analysis for image-method setup
%
% [RT_MED,RT_DATA] = ISM_RT_check(SETUP_STRUC)
% [RT_MED,RT_DATA] = ISM_RT_check(SETUP_STRUC,'arg1',val1,'arg2',val2,...)
%
% This function performs an analysis of the reverberation time that would
% result from image-source computations (using Lehmann & Johansson's
% implementation, see "Prediction of energy decay in room impulse responses
% simulated with an image-source model", J. Acoust. Soc. Am., vol. 124(1),
% pp. 269-277, July 2008) with a specific configuration defined by the
% structure SETUP_STRUC. This structure needs to contain the following
% fields:
%
%          Fs: sampling frequency (in Hz).
%        room: 1-by-3 vector of enclosure dimensions (in m), 
%              [x_length y_length z_length].
%  T20 or T60: scalar value (in s), desired reverberation time.
%           c: (optional) sound velocity (in m/s).
% abs_weights: (optional) 1-by-6 vector of absorption coefficients weights, 
%              [w_x1 w_x2 w_y1 w_y2 w_z1 w_z2].
%
% If the field SETUP_STRUC.c is undefined, the function assumes a default
% value of sound velocity of 343m/s.
%
% The field 'abs_weights' corresponds to the relative weights of each of
% the six absorption coefficients resulting from the desired reverberation
% time T60. For instance, defining 'abs_weights' as [1 1 0.8 0.8 0.6 0.6]
% will result in the absorption coefficients (alpha) for the walls in the
% y-dimension being 20% smaller compared to the x-dimension walls, whereas
% the floor and ceiling will end up with absorption coefficients 40%
% smaller (e.g., to simulate the effects of a concrete floor and ceiling).
% If this field is omitted, the parameter 'abs_weight' will default to
% [1 1 1 1 1 1], which leads to uniform absorption coefficients for all
% room boundaries.
%
% The structure SETUP_STRUC may contain one of the two fields 'T60' or
% 'T20'. This function will automatically determine which reverberation
% type is used and analyse the reverberation time accordingly. T20 is
% defined as the time required for the impulse response energy to decay
% from -5 to -25dB, whereas T60 corresponds to the time required by the
% impulse response energy to decay by 60dB. 
%
% HOWEVER: it must be noted that in case T60 is used, it is usually
%    unfeasible to simulate a room impulse response (RIR) down to -60dB in
%    practice in order to compute the RIR's exact T60 (due to numerical 
%    error and roundoff noise). Consequently, the value of T60 measured by
%    this function is instead interpolated using the initial slope of the
%    RIR's energy decay curve (EDC). Because a RIR's energy decay curve is
%    in general not exactly linear (especially for the case where the RIR
%    is computed using non-uniform absorption weights), this will typically
%    lead to discrepancies between the measured and the true T60 value in
%    the RIR, even if the environmental parameters are set to actually
%    achieve a correct T60 value. This function is therefore better suited
%    for an analysis of the reverberation time T20, which does not suffer
%    from the above drawback (EDC only required down to about -25dB). 
%
% In addition, a number of other (optional) parameters can be set using a 
% series of 'argument'--value pairs. The following parameters (arguments)
% can be used:
%
%     'PlotRes': set to 1 if plots of intermediate results are desired 
%                (execution will be paused!). Defaults to 0.
%   'NumConfig': number of source-receiver configurations to average the
%                results over. Defaults to 50. 
%  'SilentFlag': set to 1 to disable on-screen messages during execution. 
%                Defaults to 0.
%
% This function simulates a number of transfer functions (RIRs) using
% Lehmann & Johansson's implementation of the image-source method (see
% above reference), and measures the resulting ("true") reverberation time
% from the energy decay curve, computed using Schroeder's integration
% method. This process is repeated for a series of randomly selected
% source--receiver configurations to obtain a statistically representative
% set of measurements. The source and receiver positions are chosen to be
% located within the inner 80% of the room volume (away from the walls),
% and further than 0.75m from each other.
%
% This function returns 'RT_MED', the median reverberation time value (T60
% or T20) over the 'NumConfig' measurements, and 'RT_DATA', the vector of
% individual reverberation time measurements (T60 or T20). If 'SilentFlag'
% is set to 0, the function also reports the results of the reverberation
% time analysis on screen in the Matlab command window.

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

% User input variables:
VarList = {'PlotRes'        0;        % set to 1 if plots of intermediate results desired
           'NumConfig'      50;       % total number of src-rcv configurations considered
           'SilentFlag'     0};       % set to 1 to disable on-screen messages during execution
eval(SetUserVars(VarList,varargin));  % set user-definable variables

Fs = setupstruc.Fs;
room = setupstruc.room;
if isfield(setupstruc,'abs_weights'),
    weights = setupstruc.abs_weights;
else
    weights = ones(1,6);
end
if isfield(setupstruc,'c'),
    cc = setupstruc.c;
else
    cc = 343;
end
if isfield(setupstruc,'T60'),
    warning(['The function ''ISM_RT_check'' cannot practically measure the true T60 \n' ...
             'value in room impulse responses. Instead, it measures T60 by interpolating the \n' ...
             'initial slope of the RIR''s energy decay, which might not deliver accurate \n' ...
             'measurements. Please refer to ISM_RT_check''s online help for more information.%s'],'');
    rtstr = 'T60'; rtval = setupstruc.T60;
    T60 = setupstruc.T60;
    alpha = ISM_AbsCoeff('t60',T60,room,weights,'LehmannJohansson','c',cc);
else
    rtstr = 'T20'; rtval = setupstruc.T20;
    T20 = setupstruc.T20;
    T60 = [];
    alpha = ISM_AbsCoeff('t20',T20,room,weights,'LehmannJohansson','c',cc);
end
beta = sqrt(1-alpha);

RTvec = zeros(1,NumConfig);
if ~SilentFlag, PrintLoopPCw(' [ISM_RT_check] Computing sample TFs (this may take a while!). '); end
for ii=1:NumConfig,
    if ~SilentFlag, PrintLoopPCw(ii,NumConfig); end
    
    %-=:=- Select source/receiver points -=:=- 
    X_src = rand(1,3).*room*.8 + .1*room;   % avoid positions close to walls
    X_rcv = rand(1,3).*room*.8 + .1*room;
    while norm(X_src-X_rcv)<.75,    % choose new points if they are too close to each other
        X_src = rand(1,3).*room*.8 + .1*room;
        X_rcv = rand(1,3).*room*.8 + .1*room;
    end
    
    %-=:=- Compute TF -=:=- 
    TFcoeffs = ISM_RoomResp(Fs,beta,rtstr,rtval,X_src,X_rcv,room,'c',cc,'SilentFlag',1,'Delta_dB',50);
    TFlen = length(TFcoeffs);

    %-=:=- Compute EDC -=:=-
    EDCvec = zeros(1,TFlen);
    for nn=1:TFlen,
        EDCvec(nn) = sum(TFcoeffs(nn:end).^2);  % Energy decay using Schroeder's integration method
    end
    EDCvec = EDCvec/EDCvec(1);
    EDCvec(EDCvec==0) = eps;
    EDCvec = 10*log10(EDCvec);          % Decay curve in dB.

    %-=:=- Measure RT -=:=-
    timevec = [0:TFlen-1]/Fs;
    if isempty(T60),    % measure T20
        sind = find(EDCvec>=-5,1,'last');
        deltaX = timevec(2)-timevec(1);
        deltaY = EDCvec(sind+1)-EDCvec(sind);
        deltaA = -5 - EDCvec(sind);
        t1 = timevec(sind) + deltaA*deltaX/deltaY;
        sind = find(EDCvec>=-25,1,'last');
        deltaY = EDCvec(sind+1)-EDCvec(sind);
        deltaA = -25 - EDCvec(sind);
        t2 = timevec(sind) + deltaA*deltaX/deltaY;
        RTvec(ii) = t2-t1;
    else                % measure T60
        dp_del = norm(X_src-X_rcv)/cc;      % measure slope after direct path
        intstart = find(timevec<=dp_del,1,'last');
        intstop = find(EDCvec>=-45,1,'last');            % find best slope up to -45dB in the EDC
        numpts = intstop-intstart+1;
        if numpts<=1,
            error('Problem encountered while measuring T60.');
        end
        
        slopevec = NaN*ones(2,TFlen);
        errorvec = NaN*ones(1,TFlen);
        for kk=intstart+1:intstop,
            polyp = polyfit([intstart:kk],EDCvec(intstart:kk),1);   % Fit decay line on considered part of the energy decay curve
            slopevec(:,kk) = polyp.';                         % Slope of the decay line, i.e. T60 value
            errorvec(kk) = median((EDCvec - polyval(polyp,[1:TFlen])).^2);    % Fitting error for current slope based on entire decay curve
        end                                                             % using the error median avoids influence of end part of the curve
        foo = find(errorvec==nanmin(errorvec),1,'last');     % T60 estimate as decay line minimising squared error
        RTvec(ii) = 60/abs(slopevec(1,foo))/Fs;     % T60 estimate from TF computations
        polyp = slopevec(:,foo).';
        slopeind = foo;
    end

    %-=:=- Plot results if necessary -=:=-
    if PlotRes,
        reusefig('ISM Reverberation Time Analysis (ISM_RT_check.m)'); clf;
        ylimmin = -60;
        foo = TFcoeffs/sum(TFcoeffs.^2);
        plot(timevec,10*log10(foo.^2),'color',[.85 .85 1]); hold on;
        plot(timevec,EDCvec,'k');
        xlabel('time (s)'); ylabel('EDC (dB)');
        if isempty(T60),    % measure T20
            plot([0 t1 t1],[-5 -5 ylimmin],'k:');
            plot([0 t2 t2],[-25 -25 ylimmin],'k:');
            tstr = ['TF #' num2str(ii) ' of ' num2str(NumConfig) ': measured T_{20} = ' num2str(RTvec(ii),'%.3f') 's'];
        else                % measure T60
            plot(timevec([1 end]),polyval(polyp,[1 TFlen]),'r');
            plot(timevec(slopeind),polyval(polyp,slopeind),'color','r','marker','o','markersize',4);
            plot(timevec(intstart),polyval(polyp,intstart),'color','r','marker','o','markersize',4);
            tstr = ['TF #' num2str(ii) ' of ' num2str(NumConfig) ': measured T_{60} = ' num2str(RTvec(ii),'%.3f') 's'];
        end
        axis tight; ylim([ylimmin 5]);
        if ii<NumConfig, 
            title([tstr ' (hit any key to continue)']);
            pause; title(tstr); shg; pause(.001);
        else
            title(tstr);
        end
    end

end
 
medRT = median(RTvec);
iqrRT = iqr(RTvec);

%-=:=- Output results on screen -=:=-
if ~SilentFlag,
    fprintf('\n -=:=- -=:=- Results from reverberation time analysis (''ISM_RT_check.m'') -=:=- -=:=-');
    fprintf('\n Simulation parameters:  Fs = %.5gHz,  c = %.5gm/s,  room = [%.5gm %.5gm %.5gm]',Fs,cc,room(1),room(2),room(3));
    fprintf('\n                         abs_weights = [%.5g %.5g %.5g %.5g %.5g %.5g]',weights(1),weights(2),weights(3),weights(4),weights(5),weights(6));
    if isempty(T60),
        fprintf('\n Desired reverb. time:   T20 = %.5gs',T20);
    else
        fprintf('\n Desired reverb. time:   T60 = %.5gs',T60);
    end
    fprintf('\n Reflection coeff. for this setup (determined by Lehmann & Johansson''s method):')
    fprintf('\n     beta = [%.5g  %.5g  %.5g  %.5g  %.5g  %.5g]',beta(1),beta(2),beta(3),beta(4),beta(5),beta(6));
    fprintf(['\n\n Resulting ' rtstr ' measured in room (median value over %d configurations):'],NumConfig);
    fprintf(['\n     ' rtstr ' = %.5f s (inter-quartile range: %.5g s)'],medRT,iqrRT);
    if ~isempty(T60),
        fprintf(' ~~~ [T60 measurements may be inaccurate!]');
    end
    fprintf('\n -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=- -=:=-\n\n');
end
