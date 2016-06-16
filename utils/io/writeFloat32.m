% read the NIST format wav files
% Author: Xiao Xiong
% Created: 1 Feb 2005
% Last modified: 1 Feb 2005

function writeFloat32(file_name, data)
fid = fopen(file_name, 'w');
fwrite(fid, data, 'float32');
fclose(fid);