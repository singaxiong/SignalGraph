
function [grp_phase, cep] = modified_group_delay_feature_original(speech, fs, rho, gamma, num_coeff)

%input: 
%     file_name: path for the waveform. The waveform should have a header
%     rho: a parameter to control the shape of modified group delay spectra
%     gamma: a parameter to control the shape of the modified group delay spectra
%     num_coeff: the desired feature dimension
%
%output:
%     grp_phase: modifed gropu delay spectrogram
%     cep: modified group delay cepstral feature.
%
%%Example: [grp_phase, cep] = modified_group_delay_feature('./100001.wav', 0.4, 0.9, 12);
% Please tune rho and gamma for better performance
%
% by Zhizheng Wu (zhizheng.wu@ed.ac.uk)
% http://www.zhizheng.org
%
% The code has been used in the following three papers:
% Zhizheng Wu, Xiong Xiao, Eng Siong Chng, Haizhou Li, "Synthetic speech detection using temporal modulation feature", IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2013.
% Zhizheng Wu, Tomi Kinnunen, Eng Siong Chng, Haizhou Li, Eliathamby Ambikairajah, "A study on spoofing attack in state-of-the-art speaker verification: the telephone speech case", Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) 2012. 
% Zhizheng Wu, Eng Siong Chng, Haizhou Li, "Detecting Converted Speech and Natural Speech for anti-Spoofing Attack in Speaker Recognition", Interspeech 2012. 
%
% feel free to modify the code and welcome to cite above papers :)

% [speech,fs]  = wavread(file_name);

% by default, should use rho=1.2 and gamma=0.4;

frame_length = 25; %msec

frame_shift  = 10; %msec
NFFT         = 512;
pre_emph     = true;

%%% Pre-emphasis + framing 
if (pre_emph)
    speech = filter([1 -0.97], 1, speech);
end;
frame_length = round((frame_length/1000)*fs);
frame_shift = round((frame_shift/1000)*fs);
frames = enframe(speech, hamming(frame_length), frame_shift);

frame_num    = size(frames, 1);
frame_length = size(frames, 2);
delay_vector = [1:1:frame_length];
delay_matrix = repmat(delay_vector, frame_num, 1);

delay_frames = frames .* delay_matrix;

x_spec = fft(frames', NFFT);
y_spec = fft(delay_frames', NFFT);
x_spec = x_spec(1:NFFT/2+1, :);
y_spec = y_spec(1:NFFT/2+1, :);

temp_x_spec = abs(x_spec);
min_value = min(temp_x_spec(temp_x_spec > 0.0));
temp_x_spec(temp_x_spec <= 0.0) = min_value * 0.1;

dct_spec = dct(medfilt1(log(temp_x_spec), 5));
smooth_spec = idct(dct_spec(1:30,:), NFFT/2+1);

grp_phase1 = (real(x_spec).*real(y_spec) + imag(y_spec) .* imag(x_spec)) ./(exp(smooth_spec).^ (2*rho));
grp_phase = (grp_phase1 ./ abs(grp_phase1)) .* (abs(grp_phase1).^ gamma);
grp_phase = grp_phase ./ (max(max(abs(grp_phase))));

grp_phase(isnan(grp_phase)) = 0.0;

cep = dct(grp_phase);
cep = cep(2:num_coeff+1, :)';

