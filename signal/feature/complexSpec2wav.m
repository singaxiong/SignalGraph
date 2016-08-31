
% len is frame size, len1 is overlap
function wav = complexSpec2wav(complex_x, framelen, overlap)
nFFT = size(complex_x,2)*2-2;
img = sqrt(-1);

complex_x(:,nFFT/2+2:nFFT) = conj(complex_x(:,nFFT/2:-1:2));
xi = ifft(complex_x');
xi = real(xi);
wav = my_ola(xi, framelen, overlap);

A = [1 -0.97]; 
wav = filter(1, A, wav);
end