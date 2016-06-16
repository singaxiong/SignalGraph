% read the NIST format wav files
% Author: Xiao Xiong
% Created: 1 Feb 2005
% Last modified: 1 Feb 2005

function [data] = readFloat32(file_name, big_endian)
if nargin<2
    big_endian = 0;
end
FILE = fopen( file_name );
if FILE < 0
    fprintf('File open error: %s\n', file_name);
    return;
end
if big_endian
    data = fread(FILE, 'float32','ieee-be');
else
    data = fread(FILE, 'float32','ieee-le');
end
data = double(data);
%plot(data);
fclose(FILE);