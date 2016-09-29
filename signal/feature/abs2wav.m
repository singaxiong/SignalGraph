% len is frame size in samples, 
% len1 is frame overlap in samples
% mag_x is the magnitude (no log)
% phase_x is the phase returned by angle(x), where x is the complex Fourier
% coefficients. 
function wav = abs2wav(mag_x, phase_x, len, len1)
nFFT = size(mag_x,2)*2-2;
img = sqrt(-1);

x_phase = mag_x .* ( cos(phase_x) + img*sin(phase_x));
x_phase = conj(x_phase);
x_phase(:,nFFT/2+2:nFFT) = conj(x_phase(:,nFFT/2:-1:2));
xi = ifft(x_phase');
xi = real(xi);
wav = my_ola(xi, len, len1);

A = [1 -0.97]; 
wav = filter(1, A, wav);
end