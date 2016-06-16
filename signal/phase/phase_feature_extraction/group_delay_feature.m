function GDD_phase = group_delay_feature(sp_complex)

%input: 
%     sp_complex: STFT of target waveform x(n) which contain both magnitude and phase information
%
%output:
%     GDD_phase: group delay spectrogram

real_sp = real(sp_complex);
image_sp = imag(sp_complex);

diff_real_sp = real_sp(:,2:end) - real_sp(:,1:end-1);
diff_real_sp = [real_sp(:,1) diff_real_sp];

diff_image_sp = image_sp(:,2:end) - image_sp(:,1:end-1);
diff_image_sp = [image_sp(:,1) diff_image_sp];

% [diff_real_sp] = diff(real_sp);
% [diff_image_sp] = diff(image_sp);

GDD_phase = -(real_sp.*diff_image_sp - image_sp.*diff_real_sp) ./ (abs(sp_complex).^2);