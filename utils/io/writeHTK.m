% write the HTK format header
% Author: Xiao Xiong
% Created: 29 Jan 2005 
% Last modified: 29 Jan 2005

function [success] = writeHTK(output_file_name, data,para_type, big_endian)
if nargin < 4
    endian = 'ieee-le';
elseif big_endian == 1
    endian = 'ieee-be';
else
    endian = 'ieee-le';
end

[N_vector, N_spectral] = size(data);

%%%%%%%%%%%%%%%% define HTK format file header %%%%%%%%%%%%%%
hdr.nSamples   = N_vector;  % number of vectors
hdr.sampPeriod = 100000;    % sampling period 10000.0 micro seconds
hdr.sampSize   = N_spectral*4;% number of byte per vector
switch para_type
    case 'MFCC_0'
        hdr.parmKind   = 8198;      % a number represents type MFCC_0
    case 'MFCC_0_E'
        hdr.parmKind   = 8262;      % a number represents type MFCC_E_0
    case 'MFCC_0_D_A'
        hdr.parmKind   = 8966;      % a number represents type MFCC_0_D_A
    case 'MFCC_0_D_A_Z'
        hdr.parmKind   = 11014;      % a number represents type MFCC_0_D_A_Z
    case 'MFCC_E_D_A'
        hdr.parmKind   = 838;
    case 'DISCRETE'
        hdr.parmKind    = 10;
        hdr.sampSize   = N_spectral*2;% number of byte per vector
    otherwise
        error('writeHTK: Unrecognized parameter type');
end

OUTPUT = fopen(output_file_name, 'w');
fwrite(OUTPUT, hdr.nSamples, 'int32', 0, endian);
fwrite(OUTPUT, hdr.sampPeriod, 'int32', 0, endian);
fwrite(OUTPUT, hdr.sampSize, 'int16', 0, endian);
fwrite(OUTPUT, hdr.parmKind, 'int16', 0, endian);

% for i=1:N_vector
%     for j=1:N_spectral
%         fwrite(OUTPUT, data(i,j), 'float32');
%     end
% end
% for i=1:N_vector        % less for loop, faster run time
%     fwrite(OUTPUT, data(i,:), 'float32');
% end
if strcmp(para_type, 'DISCRETE')
    fwrite(OUTPUT, data', 'int16', 0, endian);
else
    fwrite(OUTPUT, data', 'float32', 0, endian);
end

fclose(OUTPUT);
success = 1;