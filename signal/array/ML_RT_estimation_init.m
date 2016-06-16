function rte_handle = ML_RT_estimation_init(fs)
%--------------------------------------------------------------------------
% Initialization for ML_RT_estimation.m
%--------------------------------------------------------------------------
%
% rte_handle = ML_RT_estimation_init(fs)
% provides struct with all parameters and buffers needed for the function
% ML_RT_estimation.m to perform a blind RT estimation by frame-wise
% processing
%
% Input
% fs        : sampling frequency in Hz
%
% Output
% rte_handle: struct with all parameters and variables
%
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% Reference:
% Heinrich W. Löllmann, Emre Yilmaz, Marco Jeub and Peter Vary:
% "An Improved Algorithm for Blind Reverberation Time Estimation"
% International Workshop on Acoustic Echo and Noise Control (IWAENC),
% Tel Aviv, Israel, August 2010.
%
% Note:
% The approach for a fast tracking of changing RTs by means of a second
% histogram is not implemented to furhter reduce the complexity of the
% algorithm. The parameter settings of this function are not the same as
% those used for the simulation examples of the reference paper.
%
% The algorithm allows to estimate the RT within a range of 0.2s to 1.2s
% and assumes that source and receiver are not within the critical
% distance.
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

% general parameters
rte_handle.fs = fs;          % sampling frequency
no = rte_handle.fs / 24e3 ;  % correction factor to account for different sampling frequency

if fs<8e3 || fs>24e3
    warning('Algorithm has not been tested for this sampling frequency!')
end

% pararmeters for pre-selection of suitable segments
if fs>8e3
    rte_handle.down = 2;                               % rate for downsampling before RT estimation to reduce computational complexity
else
    rte_handle.down = 1;
end
rte_handle.N_sub = round( no * 820/rte_handle.down);   % sub-frame length (after downsampling)
rte_handle.N_shift = round(rte_handle.N_sub*rte_handle.down/4); % frame shift (before downsampling)
rte_handle.nos_min = 3;                                % minimal number of subframes to detect a sound decay
rte_handle.nos_max = 7;                                % maximal number of subframes to detect a sound decay
rte_handle.N = rte_handle.nos_max*rte_handle.N_sub;    % maximal frame length (after downsampling)

% parameters for ML-estimation
Tmax = 1.2;                  % max RT being considered
Tmin = 0.2;                  % min RT being considered
rte_handle.bin = 0.1;                                  % step-size for RT estimation
rte_handle.Tquant = ( Tmin : rte_handle.bin : Tmax );  % set of qunatized RTs considered for maximum search
rte_handle.a = exp( - 3*log(10) ./ ( (rte_handle.Tquant) .* (rte_handle.fs/rte_handle.down)));   % corresponding decay rate factors
rte_handle.La = length( rte_handle.a );                % no. of considered decay rate factors (= no of. RTs)

% paramters for histogram-based approach to reduce outliers (order statistics)
rte_handle.buffer_size = round( no*1200/rte_handle.down); % buffer size
rte_handle.buffer = zeros( 1, rte_handle.buffer_size );  % buffer with previous indices to update histogram
rte_handle.no_bins  = rte_handle.La;                     % no. of histogram bins
rte_handle.hist_limits = Tmin - rte_handle.bin/2 : rte_handle.bin :  Tmax + rte_handle.bin/2 ; % limits of histogram bins
rte_handle.hist_rt = zeros(1,rte_handle.no_bins);        % histogram with ML estimates
rte_handle.hist_counter = 0;                             % counter increased if histogram is updated

% paramters for recursive smoothing of final RT estimate
rte_handle.alpha = 0.996;          % smoothing factor
rte_handle.RT_initial = 0.3;       % initial RT estimate
rte_handle.RT_last = rte_handle.RT_initial; % last RT estimate
rte_handle.RT_raw  = rte_handle.RT_initial; % raw RT estimate obtained by histogram-approach

%--------------------------------------------------------------------------