% Add default configurations to STFT
% 
% Created by Xiong Xiao, Temasek Laboratories, NTU, Singapore.
% Last Modified: 29 Jun 2016
%
function para = ConfigBasicSTFT(para)
para.topology.fs = 16000;   % sampling rate
para.topology.fft_len = 512;
para.topology.nFbank = 40;
% define the parameters for extracting Fourier coefficients
para.topology.frame_len = para.topology.fs * 0.025;
para.topology.frame_shift = para.topology.fs * 0.01;
para.topology.removeDC = 0;     % do not remove DC for faster speed. We are going to do CMN anyway
para.topology.win_type = 'hamming';
end

