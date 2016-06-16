% read the NIST format wav files
% Author: Xiao Xiong
% Created: 1 Feb 2005
% Last modified: 1 Feb 2005

function writeRaw(file_name, data, big_endian)

if nargin < 3
    big_endian = 0;
end
FILE = fopen( file_name ,'w');
if FILE < 0
    fprintf('File open error: %s\n', file_name);
    return;
end

max_abs = max(max(abs(data)));
data = data/max_abs*2^15;
data = round(data);

if big_endian
    fwrite(FILE, data, 'int16','ieee-be');
else
    fwrite(FILE, data, 'int16','ieee-le');
end
fclose(FILE);