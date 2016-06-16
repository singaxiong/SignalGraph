% write the HTK format header
% Author: Xiao Xiong
% Created: 29 Jan 2005 
% Last modified: 8 Feb 2005

function [success] = writeFBANK(output_file_name, data);

[N_vector, N_spectral] = size(data);

%%%%%%%%%%%%%%%% define HTK format file header %%%%%%%%%%%%%%
hdr.nSamples   = N_vector;  % number of vectors
hdr.sampPeriod = 100000;    % sampling period 10000.0 micro seconds
hdr.sampSize   = 80;        % 20 elements in one vector and 4 bytes per element
hdr.parmKind   = 7;         % type FBANK

OUTPUT = fopen(output_file_name, 'w');
fwrite(OUTPUT, hdr.nSamples, 'int32');
fwrite(OUTPUT, hdr.sampPeriod, 'int32');
fwrite(OUTPUT, hdr.sampSize, 'int16');
fwrite(OUTPUT, hdr.parmKind, 'int16');

for i=1:N_vector
    for j=1:N_spectral
        fwrite(OUTPUT, data(i,j), 'float32');
    end
end
fclose(OUTPUT);
success = 1;