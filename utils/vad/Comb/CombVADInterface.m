function vad = CombVADInterface(wavfile, para, DEBUG)
if nargin<3
    DEBUG = 0;
end
if isfield(para, 'isWavFile') == 0
    para.isWavFile = 1;
else
    if isfield(para,'fs')==0
        Fs = 16000;
    else
        Fs = para.fs;
    end
end
if isfield(para,'frame_rate')==0
    para.frame_rate = 100;
end
if isfield(para, 'buffer_len')==0
    para.buffer_len = 40;
end

if para.isWavFile
    [wav, Fs] = audioread(wavfile);
else
    wav = wavfile;
end
[comb_vad_flag, energy_val, periodicVal] = CombVAD(wav, Fs, DEBUG);
currVAD = comb_vad_flag;

% interpolating the vad values. 
% The comb vad uses a frame rate of 50Hz. We need to do interpolation to
% get VAD, if the desired frame rate is different from 50Hz. 
comb_idx = 0.05:0.02:(length(currVAD)*0.02+0.03);
desired_idx = (1/para.frame_rate) : (1/para.frame_rate) : comb_idx(end);
currVAD = interp1(comb_idx, currVAD, desired_idx, 'nearest', 'extrap');

% Post process VAD with a buffer length of 0.4s
[vad, vad_extended] = PostProcessVAD(currVAD, para.buffer_len*para.frame_rate/100);

end