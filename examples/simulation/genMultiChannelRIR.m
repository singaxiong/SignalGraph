function RIR = genMultiChannelRIR(t60, useGPU, filename)
if nargin<2
    useGPU = 0;
end

% the following geometry parameters can be randomly sampled from a
% distribution. For example, you want to cover different room sizes, just
% sample the length, width, and height of the room with a reasonable
% distribution. 
nCh = 8;    % number of microphones in the array
room_size = [10 5 3];   % length, width, and height of the room
ctr_pos = [2 2 1];      % center position of the array
src_pos = [5 3 2.5];    % coordinate of the sound source
[mic_pos] = genMicPositionsWrtCenter(0);        % get the microphones' positions with respect to the center of the array
mic_pos = bsxfun(@plus, mic_pos, ctr_pos(:));   % get microphones' positions

% set up the settings
SetupStruc.Fs = 16000;                  % sampling frequency in Hz
SetupStruc.c = 343;                     % (optional) propagation speed of acoustic waves in m/s
SetupStruc.T60 = t60;                   % reverberation time T60, or define a T20 field instead!
SetupStruc.room = room_size;
SetupStruc.reflect_weights = ones(1,6) * exp(-13.82/sum(1./SetupStruc.room(1,:)*SetupStruc.c*t60));     %calculate the reflection coefficient. 
                                        % It's good to first decide the T60 and room size, and then compute the corresponding to reflection rate. 
SetupStruc.mic_pos = mic_pos';
SetupStruc.src_traj = src_pos;

% we are using a modified version of the ISM_RIR_bank function from Eric A. Lehmann
% The main change is that it now supports the use of GPU. 
[RIR_cell] = ISM_RIR_bank_GPU(SetupStruc,'', 'useGPU', useGPU, 'SilentFlag', 1);     % generating RIR can be very slow for long T60. Using GPU makes it much faster
RIR = cell2mat(RIR_cell')';

if nargin==3
    wavReader = BinaryReader(nCh, 'int16');                 % a binary read/write object
    wavReader.write(filename, StoreWavInt16(RIR));          % we can write the RIR into binary format for later access
end

end
