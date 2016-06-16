function [ RT, par ] = ML_RT_estimation_frame( frame, par )
%--------------------------------------------------------------------------
% RT estimation by frame-wise processing
% -------------------------------------------------------------------------
%
% [ RT, par ] = ML_RT_estimation_frame( frame, par )
% performs blind estimation of the reverberation time (RT) by frame-wise
% processing
%
% Input
% frame : segment with reverberant (denoised) speech of length par.N_sub
%
% par   : struct with all parameters and buffers created by the function
%         ML_RT_estimation_init.m to enable frame-wise processing
%
% Output
% RT    : estimated RT
%
% par   : struct with updated buffers to enable frame-wise processing as
%         well as intermediate values
%
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% Reference:
% Heinrich W. Löllmann, Emre Yilmaz, Marco Jeub and Peter Vary:
% "An Improved Algorithm for Blind Reverberation Time Estimation"
% International Workshop on Acoustic Echo and Noise Control (IWAENC),
% Tel Aviv, Israel, August 2010.
% (availabel at www.ind.rwth-aachen.de/~bib/loellmann10a)
%
% Note:
% The approach for a fast tracking of changing RTs by means of a second
% histogram is not implemented to furhter reduce the complexity of the
% algorithm.
%
% The algorithm allows to estimate the RT within a range of 0.2s to 1.2s
% and assumes that source and receiver are not within the critical
% distance. A denoising is not performed by this function and has to be
% done in advance.
%
%--------------------------------------------------------------------------
% Copyright (c) 2012, Heinrich Loellmann and Marco Jeub
% Institute of Communication Systems and Data Processing
% RWTH Aachen University, Germany
% Contact information: loellmann@ind.rwth-aachen.de
%
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the RWTH Aachen University nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%--------------------------------------------------------------------------
% Version 1.0
%--------------------------------------------------------------------------

if length(frame) < par.N_sub
    error('input frame too short')
end

[ M , N ] = size( frame );
if M>N
    h = frame.';
else
    h = frame;
end
% ensures a column vector


cnt = 0;     % sub-frame counter for pre-selection of possible sound decay
RTml = -1;   % default RT estimate (-1 indicates no new RT estimate)

% calculate variance, minimum and maximum of first sub-frame
seg = frame( 1 : par.N_sub );

var_pre = var( seg );
min_pre = min( seg );
max_pre = max( seg );

for k = 2 : par.nos_max,
    
    % calculate variance, minimum and maximum of succeding sub-frame
    seg = frame( 1+(k-1)*par.N_sub : k*par.N_sub );
    var_cur = var( seg );
    max_cur = max( seg );
    min_cur = min( seg );
    
    % -- Pre-Selection of suitable speech decays --------------------
    
    if (var_pre > var_cur) && (max_pre > max_cur) && (min_pre < min_cur)
        % if variance, maximum decraease and minimum increase
        % => possible sound decay detected
        
        cnt = cnt + 1;
        
        % current values becomes previous values
        var_pre = var_cur;
        max_pre = max_cur;
        min_pre = min_cur;
        
    else
        
        if cnt >= par.nos_min % minimum length for assumed sound decay achieved?
            
            % -- Maximum Likelihood (ML) Estimation of the RT
            RTml = max_loglf( frame(1:cnt*par.N_sub), par.a, par.Tquant);
            
        end
        
        break
        
    end
    
    if k == par.nos_max % maximum frame length achieved?
        
        RTml = max_loglf( frame(1:cnt*par.N_sub), par.a, par.Tquant );
        
    end
    
end % eof sub-frame loop


if RTml >= 0  % new ML estimate calculated
    
    % apply order statistics to reduce outliers
    par.hist_counter = par.hist_counter+1;
    
    for i = 1: par.no_bins,
        
        % find index corresponding to the ML estimate
        if  ( RTml >= par.hist_limits(i) ) && ( RTml <= par.hist_limits(i+1) )
            
            index = i;
            break
        end
    end
    
    % update histogram with ML estimates for the RT
    par.hist_rt( index ) = par.hist_rt( index ) + 1;
    
    if par.hist_counter > par.buffer_size +1
        % remove old values from histogram
        par.hist_rt( par.buffer( 1 ) ) = par.hist_rt( par.buffer( 1 ) ) - 1;
    end
    
    par.buffer = [ par.buffer(2:end), index ]; % update buffer with indices
    [ dummy, idx ] = max( par.hist_rt );       % find index for maximum of the histogram
    
    par.RT_raw = par.Tquant( idx );   % map index to RT value
    
end

% final RT estimate obtained by recursive smoothing
RT = par.alpha * par.RT_last + (1-par.alpha) * par.RT_raw;
par.RT_last = RT;

par.RTml = RTml;    % intermediate ML estimate for later analysis


return

%--------------------------------------------------------------------------
function [ ML, ll ] = max_loglf(h, a, Tquant)
%--------------------------------------------------------------------------
% [ ML, ll ] = max_loglf( h, a, Tquant )
%
% returns the maximum of the log-likelihood (LL) function and the LL
% function itself for a finite set of decay rates
%
% Input arguments
% h: input frame
% a: finite set of values for which the max. should be found
% T: corresponding RT values for vector a
%
% Output arguments
% ML : ML estimate for the RT
% ll : underlying LL-function


N = length(h);
n = (0:N-1); % indices for input vector
ll = zeros(length(a),1);

h_square = (h.^2).';

for i=1:length(a),
    
    Sum  = ( a(i).^(-2*n) ) * h_square ;
    
    if Sum < 1e-12
        ll( i ) = -inf;
    else
        ll( i ) = -N/2*( (N-1)*log( a(i) ) + log( 2*pi/N * Sum ) + 1 );
    end
    
end

[ dummy, idx ] = max( ll ); % maximum of the log-likelihood function
ML = Tquant( idx );         % corresponding ML estimate for the RT


return
%--------------------------------------------------------------------------