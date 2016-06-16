% read the NIST format wav files
% Author: Xiao Xiong
% Created: 1 Feb 2005
% Last modified: 1 Feb 2005

function [data] = readNIST(file_name,big_endian);

FILE = fopen( file_name );
if FILE <1
    error('File open failed: %s', file_name);
end
fread(FILE, 1024, 'int8');      %read the header first
if big_endian
    data = fread(FILE, 'int16=>short','b');
else
    data = fread(FILE, 'int16=>short','l');
end
% plot(data);
data = double(data(1:length(data)));        % store the vecors as column vector of wav
% plot(data);
fclose(FILE);

