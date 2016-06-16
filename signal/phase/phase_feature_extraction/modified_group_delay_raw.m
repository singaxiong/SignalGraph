
function grp_phase = modified_group_delay_raw(sp_complex, sp_delay)

%input: 
%     sp_complex: STFT of target waveform x(n) which contain both magnitude and phase information
%     sp_delay: STFT of modified target waveform n*x(n) 
%
%output:
%     grp_phase: modifed group delay spectrogram

x_spec = sp_complex';
y_spec = sp_delay';

smooth_len = size(y_spec, 1);

dct_spec = dct(medfilt1(log(abs(x_spec) + 0.000000001), 5));
smooth_spec = idct(dct_spec(1:30,:), smooth_len);

grp_phase = (real(x_spec).*real(y_spec) + imag(y_spec) .* imag(x_spec)) ./(exp(smooth_spec).^ (2));
grp_phase = grp_phase';


