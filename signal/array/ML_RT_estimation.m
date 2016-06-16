function [rt_est,rt_est_mean, RT_est, rte_handle ] = ML_RT_estimation(x,simpar)
%--------------------------------------------------------------------------
% Blind RT estimation by means of a maximum-liklihood (ML) estimator
%--------------------------------------------------------------------------
%
%  [rt_est,rt_est_mean, RT_est, rte_handle ] = ML_RT_estimation(x,simpar)
%  provides RT estimates over time for the input sequence x using a
%  frame-wise processing scheme. The estimation algorithm is described in
%  the reference below.
%
% Input
%   x:  reverberant speech
%       (for noisy speech, a denoising has to be performed before)
%
%  simpar: struct containing
%      simpar.block_size : block size of overall processing scheme
%      simpar.overlap    : frame shift (overlap) of overall processing
%                      scheme
%      simpar.fs         : sampling frequency in Hz
%
%
% Output
%   rt_est      : estimated RT (T60) over time where the time interval for
%                 the estimates is given by rte_handle.N_shift/simpar.fs
%
%   rt_est_mean : mean RT estimate determined from rt_est
%
%   RT_est      : estimated RT (T60) over time where the time interval for
%                 the estimates is given by simpar.overlap/simpar.fs
%
%   rte_handle  : struct containing all buffers, variables and parameters
%                 used for frame-wise RT estimation
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
%
% Reference:
% Heinrich W. Löllmann, Emre Yilmaz, Marco Jeub and Peter Vary:
% "An Improved Algorithm for Blind Reverberation Time Estimation"
% International Workshop on Acoustic Echo and Noise Control (IWAENC),
% Tel Aviv, Israel, August 2010.
% (availabel at www.ind.rwth-aachen.de/~bib/loellmann10a)
%
% Note:
% The approach for a fast tracking of changing RTs by means of a second
% histogram is not implemented in this version to furhter reduce the
% complexity of the algorithm.
%
% The algorithm allows to estimate the RT within a range of 0.2s to 1.2s
% and assumes that source and receiver are not within the critical
% distance. A denoising is not performed by this program and has to be
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

%--------------------------------------------------------------------------
% Check input parameters
%--------------------------------------------------------------------------
[num_ch,num_samples] = size(x);

if nargin < 2
    error('not enough input parameters');
end

if (num_ch ~= 1)
    error('Input signals must be single channel vectors [1xN]');
end


%--------------------------------------------------------------------------
% Get parameters
%--------------------------------------------------------------------------

block_size = simpar.block_size; % to ease notation only
overlap = simpar.overlap;
fs = simpar.fs;

% if ~isfield(simpar,'remove_from_avg')
%     remove_from_avg = [1,1];
%     % first and last values are not taken into account for averaging;
%     % given in seconds !
% else
%     remove_from_avg = simpar.remove_from_avg;
% end
% 
% if num_samples/fs < remove_from_avg(1)
%     error('remove_from_avg(1) larger than given speech signal');
% end


%--------------------------------------------------------------------------
% Initialize RT estimation
%--------------------------------------------------------------------------
rte_handle = ML_RT_estimation_init(fs);

%--------------------------------------------------------------------------
% Block processing
%--------------------------------------------------------------------------

% Note
% Block size and overlap are not indentical to the block sizes and block
% shifts given by rte_handle to demonstrate the integration of the RT
% estimation into a processing scheme with a given block size and frame shift
% (including sample wise processing as special case).

% vectors for all RT estimates
rt_est = ones( ceil(length(x)/rte_handle.N_shift)+1, 1)*rte_handle.RT_initial;
RT_est = zeros( ceil(length(x)/simpar.overlap)+1, 1);

% initialize counters
k = 0;
rt_frame_cnt = 0;
RT = rte_handle.RT_initial;
k_rt = rte_handle.N*rte_handle.down +1;    % index counter for RT estimation

for cnt = 1:overlap:num_samples-block_size+1
    k = k+1;  % frame counter for overall block processing scheme
    
    %----------------------------------------------------------------------
    % New T60 Estimation
    %----------------------------------------------------------------------
    if cnt > k_rt
        rt_frame_cnt = rt_frame_cnt + 1; % frame counter for RT estimation
        
        x_seg = x( k_rt - rte_handle.N*rte_handle.down + 1 : rte_handle.down : k_rt );
        [RT, rte_handle] =  ML_RT_estimation_frame( x_seg, rte_handle ); % actual RT estimation
        k_rt = k_rt + rte_handle.N_shift;  % increase index counter for RT estimation
        
        rt_est(rt_frame_cnt) = RT;% save RT estimate over time
    end
    %----------------------------------------------------------------------
    
    RT_est( k ) = RT;
end

RT_est = RT_est(1:k);
rt_est = rt_est(1:rt_frame_cnt);
rt_est_mean = mean(rt_est);
%--------------------------------------------------------------------------
% Mean RT, averaged over all considered frames
%--------------------------------------------------------------------------
% fr2sec_idx = linspace(1,num_samples/fs,rt_frame_cnt);
% idx_tmp = find(fr2sec_idx > remove_from_avg(1));
% if isempty(idx_tmp)
%     error('input signal is too short for given frame removal');
% end
% idx(1) = idx_tmp(1);
% idx_tmp = find(fr2sec_idx < (fr2sec_idx(end)-remove_from_avg(2)));
% idx(2) = idx_tmp(end);
% avg_range = idx(1):idx(2);

% rt_est_mean = mean(rt_est(avg_range));

