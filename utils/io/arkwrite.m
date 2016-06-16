function [status] = arkwrite(filename_ark_out, HEADER_MAT, FEATURE_MAT)
% [status] = arkwrite(filename_ark_out, HEADER_MAT, FEATURE_MAT)
%
% Stores features in the Kaldi ark feature file based on the information provided in HEADER_MAT and FEATURE_MAT.
% Generates .scp file with the list of all 'utterance' files stored in the ark file.
%
% - filename_ark_out - output ark file name
%
% - HEADER_MAT contains headers of the individual tokens, each row capturing one token.
% The row contains token file name, number of feature frames, number of frame dimensions, and start and
% end row index in the FEATURE_MAT matrix where the features of this token are stored.
%
% - FEATURE_MAT contains concatenated raw feature vectors from all tokens. The rows represent individual
% feature frames and columns their dimensions.
%
% Example:
%
% HEADER_MAT = 
%
%   'MDAB0_SI1039'    [392]    [13]    [   1]    [ 392]
%   'MDAB0_SI1669'    [204]    [13]    [ 393]    [ 596]
%
% The matrix reads: a token labeled MDAB0_SI1039 contains 392 frames, each frame has 13 dimension, and the feature frames are stored in rows 1-392 in FEATURE_MAT (can be accessed as FEATURE_MAT(1:932, :)).
%                   This token is followed by a token labeled MDAB0_SI1669 which contains 204 frames, etc.
%

% Copyright 2013 Hynek Boril, Center for Robust Speech Systems (CRSS), The University of Texas at Dallas
%
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%       http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.

% Contact: borilh@gmail.com

% HEADER_MAT - matrix of headers - each row contains token filename, number of frames, number of dimensions per frame, and start and end row index in the FEATURE_MAT matrix where the features will be stored 
% FEATURE_MAT - matrix of concatenated feature vectors from all tokens
FEATURE_MAT_row_counter = 1;

FID_ARK = fopen(filename_ark_out, 'w');
list_size = size(HEADER_MAT);
number_of_files = uint32(list_size(1));

fea_size = size(FEATURE_MAT);
fea_vector_length = uint32(fea_size(2));

%------- Write features into ark file ----------
for z = 1:number_of_files,
	fea_name_raw = char(HEADER_MAT(z, 1));
	fwrite(FID_ARK, fea_name_raw, 'char');
	fwrite(FID_ARK, ' ', 'char');
	fwrite(FID_ARK, 0, 'uint8');
	fwrite(FID_ARK, 'BFM ', 'char');
	fwrite(FID_ARK, 4, 'uint8');
	no_frames = cell2mat(HEADER_MAT(z, 2));
	fwrite(FID_ARK, no_frames, 'uint32');
	fwrite(FID_ARK, 4, 'uint8');
	fea_vector_length = cell2mat(HEADER_MAT(z, 3));
	fwrite(FID_ARK, fea_vector_length, 'uint32');
	fea_index_start = cell2mat(HEADER_MAT(z, 4));
	fea_index_end = cell2mat(HEADER_MAT(z, 5));
	fea_act = FEATURE_MAT(fea_index_start:fea_index_end, :)';
	fwrite(FID_ARK, fea_act(:), 'float32');
end

fclose(FID_ARK);

status = 1;
