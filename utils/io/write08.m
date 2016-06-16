% read the NIST format wav files
% Author: Xiao Xiong
% Created: 1 Feb 2005
% Last modified: 1 Feb 2005

function write08(file_name, data);

FILE = fopen( file_name, 'w' );
fwrite(FILE, data, 'int16');
fclose(FILE);