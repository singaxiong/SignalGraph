function [AuData] = ISM_AudioData(RIRFileName,SrcSignal,varargin)
%ISM_AudioData  Creates audio samples from a pre-computed RIR bank
%
% [AUDIO_DATA] = ISM_AudioData(RIR_FILE_NAME,SRC_SIGNAL)
% [AUDIO_DATA] = ISM_AudioData( ... ,'arg1',val1,'arg2',val2,...)
% 
% This function generates samples of audio data based on a bank of
% pre-computed room impulse responses (RIRs). The input variable
% RIR_FILE_NAME determines the name of the .mat file where the bank of RIRs
% is stored (the .mat file must contain a cell array parameter named
% 'RIR_cell'). The parameter RIR_FILE_NAME can be a full access path to the
% desired file. If no access path is given, the function looks for the
% desired file in the current working directory. The second input parameter
% SRC_SIGNAL corresponds to a one-dimensional vector of signal data emitted
% by the acoustic source. The length of the source audio sample, together
% with the source trajectory defined by the RIRs in RIR_FILE_NAME, define
% the velocity of the speaker across the environment (constant velocity
% motion).
%
% This function returns AUDIO_DATA, the matrix of audio data (non-
% normalised) generated at the receivers. This matrix is arranged so that
% each column represents the signal received at the corresponding sensor.
%
% The specific simulation parameters, such as room dimensions, microphone
% positions, number of microphones, source trajectory, sampling frequency,
% etc., are implicitely defined by the set of RIRs contained in
% RIR_FILE_NAME. It is hence up to the user to check the integrity of the
% resulting audio data by ensuring consistency of the considered source
% signal and the environmental setup parameters that were used to compute
% the impulse responses. For instance, the source signal defined in
% SRC_SIGNAL is assumed to have the same sampling frequency as that defined
% for the RIRs.
%
% In addition, a number of other (optional) parameters can be set using a 
% series of 'argument'--value pairs. The following parameters (arguments)
% can be used:
%
%   'AudioFileName': character string defining the name of the file used to
%                    save the resulting audio data. Can be defined as a
%                    '.mat' or '.wav' string to save the data in the
%                    corresponding format. The data is saved as a single
%                    matrix of multi-channel data, with each column
%                    containing the data generated for the corresponding
%                    microphone. If saving as a '.wav' file, the audio data
%                    is first normalised in order to avoid clipping during
%                    the wavwrite operation. 'AudioFileName' may contain a
%                    full access path to the desired file; if no access
%                    path is given, the resulting audio data is saved in
%                    the current working directory. If the file name
%                    already exists, the user will be prompted to overwrite
%                    it. If this parameter is set to the empty matrix [],
%                    no data will be saved. Defining 'AudioFileName' as a
%                    '.wav' file also requires the parameter 'Fs' to be
%                    defined (see below). Defaults to [].
%              'Fs': scalar value of sampling frequency (in Hz). Only
%                    required if the audio data is saved as a .wav file.
%                    Defaults to [].
%    'AddNoiseFlag': set this flag to 1 in order to add white Gaussian
%                    noise to the resulting sensor data. Defaults to 0.
%        'NoiseSNR': scalar value (in dB), desired SNR level of the additive
%                    Gaussian noise if 'AddNoiseFlag' is set to 1 (irrelevant
%                    otherwise). The SNR is computed as a time average
%                    across all sensors. Defaults to 20.
%         'TrajDir': direction of the source along the trajectory path. Can
%                    be defined as one of the following strings (value): 
%                      *  'SE': the source follows the path defined by the 
%                               RIRs in RIR_FILE_NAME from Start to End
%                      *  'ES': the source follows the path backwards, from 
%                               End to Start
%                      * 'SES': the source follows the path from the Start
%                               to End point, then returns back to the
%                               Start position  
%                      * 'ESE': the source follows the path backwards from 
%                               the End to the Start point, then returns 
%                               back to the End position 
%                      * 'SMS': the source follows half the path from the
%                               Start to Middle point, then returns back to
%                               the Start position 
%                      * 'EME': the source follows half the path backwards 
%                               from the End to Middle point, then returns 
%                               back to the End position 
%                    Default is 'SE'. 
%  'TruncateMicSig': due to the effects of reverberation, the resulting
%                    sensor signals will typically end up slightly longer
%                    than the source signal in SRC_SIGNAL. The parameter
%                    'TruncateMicSig' can be set to 1 in order to truncate
%                    the resulting audio data to the same length as the
%                    input signal, thereby cropping out the reverberation
%                    effects created by the room on the last few samples
%                    of input signal. Defaults to 1.
%      'SilentFlag': set to 1 to disable this function's on-screen messages.
%                    Defaults to 0.
%
% This function creates the audio data by splitting the source data in
% SRC_SIGNAL into as many (non-overlapping) frames as the number of source
% trajectory points. For a given sensor, each frame of signal is then
% convolved with the impulse response for the corresponding trajectory
% location, and the convolution results are then combined additively to
% generate the microphone signal. This process is repeated for each
% trajectory point and each microphone (overlap-add method). This leads to
% the implicit approximation that the source remains stationary during each
% frame. Also note that some clicking noise might result in the audio data
% if the RIRs differ too much from one trajectory point to the next (to a
% given sensor). It is thus up to the user to ensure that trajectory points
% are close enough to each other so as to minimise the resulting
% discrepancies between consecutive RIRs at each microphone.

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

VarList = {'AudioFileName'      [];         % file name to save audio data
           'Fs'                 [];         % sampling frequency if data to be saved as .wav file
           'AddNoiseFlag'       0;          % Set to 0 if you don't want additive noise
           'NoiseSNR'           20;         % Desired SNR in the resulting signals (in dB)
           'TrajDir'            'SE';       % direction of the source along the trajectory
           'TruncateMicSig'     1;          % truncate mic signals to same length as input signal
           'SilentFlag'         0};         % set to 1 for silent behaviour.
eval(SetUserVars(VarList,varargin));   % set user-definable variables

if min(size(SrcSignal))~=1,
    error('Source signal must be one-dimensional (single channel).');
end
SrcSignal = SrcSignal(:);   % make sure vector is column vector

if ~isempty(AudioFileName),    % check save file name
    if length(AudioFileName)<=4 || (~strcmpi(AudioFileName(end-3:end),'.mat') && ~strcmpi(AudioFileName(end-3:end),'.wav')),
        AudioFileName = [AudioFileName '.mat'];     % if not specified, save data as .mat file
    end
    if strcmpi(AudioFileName(end-3:end),'.wav') && isempty(Fs),
        error('Sampling frequency ''Fs'' must be defined to be able to save audio data as .wav file.');
    end
    if exist(AudioFileName,'file')==2,
        foo = input(' [ISM_AudioData] The audio file name already exists. Overwrite? [yes/no]: ','s');
        if ~strcmpi(foo,'yes');
            fprintf('                 Please call this function again with a different file name argument.\n');
            return
        end
    end
end

load(RIRFileName);       % load RIR bank 'RIR_cell' from file

nSamp = length(SrcSignal);      % total number of samples in the audio data
nMics = size(RIR_cell,1);       % number of mics
nFrames = size(RIR_cell,2);     % number of trajectory points

if strcmpi(TrajDir,'ES'),
    RIR_cell = RIR_cell(:,end:-1:1);
elseif strcmpi(TrajDir,'SMS'),
    RIR_cell = RIR_cell(:,[1:floor(nFrames/2) ceil(nFrames/2):-1:1]);
elseif strcmpi(TrajDir,'EME'),
    RIR_cell = RIR_cell(:,[end:-1:ceil(nFrames/2+1) floor(nFrames/2+1):end]);
elseif strcmpi(TrajDir,'SES'),
    RIR_cell = RIR_cell(:,[1:end-1 end:-1:1]);
    nFrames = 2*nFrames-1;
elseif strcmpi(TrajDir,'ESE'),
    RIR_cell = RIR_cell(:,[end:-1:1 2:end]);
    nFrames = 2*nFrames-1;
end

nSampPerFrame = ceil(nSamp/nFrames);	% total number of frames in the source signal, last frame might be partially filled

% pre-determine max number of samples in the resulting mic signals (RIR lengths are variable!):
maxEndPt = 0;
for tt=1:nFrames,
    if tt==nFrames, 
        FrameStopInd = nSamp;
    else
        FrameStopInd = tt*nSampPerFrame;
    end
    for mm=1:nMics,
        RIRlen = length(RIR_cell{mm,tt});       % length of current RIR can be variable!
        maxEndPt = max(maxEndPt,FrameStopInd+RIRlen-1);     % endpoint of last source audio sample in current frame convolved with RIR
    end
end

%-=:=- Compute audio data -=:=-
AuData = zeros(maxEndPt,nMics);
if ~SilentFlag, PrintLoopPCw(' [ISM_AudioData] Computing audio data. '); end;
for tt=1:nFrames,
   FrameStartInd = (tt-1)*nSampPerFrame+1;	% Start/end indices of the current frame in the overall audio sample.
   if tt==nFrames,
       FrameStopInd = nSamp;
   else
       FrameStopInd = tt*nSampPerFrame;
   end
   FrameData = SrcSignal(FrameStartInd:FrameStopInd);   % get one frame of data
   for mm=1:nMics,    % Compute the received signal by convolving the source signal with the RIR
      if ~SilentFlag, PrintLoopPCw((tt-1)*nMics+mm,nFrames*nMics); end;
      hh = RIR_cell{mm,tt};
      RIRlen = length(hh);       % length of current RIR (can be variable!)
      EndIndex = FrameStopInd+RIRlen-1;  % max length for the current convolution.
      AuData(FrameStartInd:EndIndex,mm) = AuData(FrameStartInd:EndIndex,mm) + freq_conv(hh,FrameData);
   end
end

%-=:=- Truncate signals -=:=-
if TruncateMicSig==1,
    AuData = AuData(1:nSamp,:);
end

%-=:=- Additive random noise -=:=-
if AddNoiseFlag==1,
    av_pow = mean( sum(AuData.^2,1)/size(AuData,1) );       % Average mic power across all received signals.
    sigma_noise = sqrt( av_pow/(10^(NoiseSNR/10)) );		% st. dev. of white noise component to achieve desired SNR.
    AuData = AuData + sigma_noise*randn(size(AuData));      % Add some random noise
end

%-=:=- Save results to file -=:=-
if ~isempty(AudioFileName),    
    if strcmpi(AudioFileName(end-3:end),'.mat') 
        save(AudioFileName,'AuData');
        if ~SilentFlag, fprintf(' [ISM_AudioData] Audio data ''AuData'' saved in file ''%s''\n',AudioFileName); end;
    else %strcmpi(AudioFileName(end-3:end),'.wav')),
        foo = max(max(abs(AuData)));
        AuData = AuData*0.99/foo;       % avoid clipping with wavwrite
        wavwrite(AuData,Fs,AudioFileName);
        if ~SilentFlag, fprintf(' [ISM_AudioData] Audio data saved in file ''%s''\n',AudioFileName); end;
    end
end
