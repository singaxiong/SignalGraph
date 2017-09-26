% read the NIST format wav files
% Author: Xiao Xiong
% Created: 1 Feb 2005
% Last modified: 1 Feb 2005

function [data] = readBool(file_name, big_endian, vecSize, idx)
if nargin<4 || isempty(idx)
    startSample = 0;
    stopSample = -1;
else
    startVecIdx = idx(1);
    stopVecIdx = idx(2);
end
if nargin<3
    vecSize = 1;
end
if nargin<2
    big_endian = 0;
end

FILE = fopen( file_name );
if FILE < 0
    fprintf('File open error: %s\n', file_name);
    data = [];
    return;
end

if big_endian
    data = fread(FILE, 'ubit1','ieee-be');
else
    data = fread(FILE, 'ubit1','ieee-le');
end
nVec = floor(length(data)/vecSize);
data = reshape(data(1:vecSize*nVec), vecSize, nVec);

fclose(FILE); 