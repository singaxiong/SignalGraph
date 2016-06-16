function [HEADER_MAT, FEATURE_MAT] = htk2ark(list_fea_htk, fname_ark_out)
% [HEADER_MAT, FEATURE_MAT] = htk2ark(list_input_htk_feature_files, filename_ark_feature_file_out)
%
% Reads htk feature files from the input list and stores all of them in single output ark file.
% Produces corresponding .scp list file.
% Generates HEADER_MAT and FEATURE_MAT that contain ark-style file information and the corresponding features.
% Requires the Voicebox function readhtk being avaliable on the system and in the path.
% Requires ark2scp() function in the path. Avalable at http://www.utdallas.edu/~hynek/tools.html

%
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
%
% Contact: borilh@gmail.com



[fnames_fea_in] = char(textread(list_fea_htk,'%s'));
number_of_files = size(fnames_fea_in);
number_of_files = number_of_files(1);

HEADER_MAT = [];       % matrix of headers - each row contains token filename, number of frames, number of dimensions per frame, and start and end row index in the FEATURE_MAT matrix where the features will be stored 
FEATURE_MAT = [];      % matrix of concatenated feature vectors from all tokens
FEATURE_MAT_row_counter = 1;

FID_ARK = fopen(fname_ark_out, 'w');

%------- Read in HTK features ----------
for z = 1:number_of_files,
	fname_fea_in = fnames_fea_in(z,:);
	[fea, frame_period_per_sec, data_type, data_type_full, data_type_ascii]  = readhtk(fname_fea_in);

	fea_size = size(fea);
	no_frames = uint32(fea_size(1));
	fea_vector_length = uint32(fea_size(2));

	if (isempty(regexp(fname_fea_in, '/')))               % test if filename contains a path
		fea_name_raw = fea_name_in;		      % if no path present, take this as the raw name
	else
		fea_name_raw = regexp(fname_fea_in, '.*\/(\w*)', 'tokens');
		fea_name_raw = fea_name_raw{1};
	end

	HEADER_MAT = [HEADER_MAT; {char(fea_name_raw)}, no_frames, fea_vector_length, FEATURE_MAT_row_counter, FEATURE_MAT_row_counter + no_frames - 1];
	FEATURE_MAT_row_counter = FEATURE_MAT_row_counter + no_frames;
	FEATURE_MAT = [FEATURE_MAT; fea];
end

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

%------- Produce corresponding scp file ----------
status = ark2scp(fname_ark_out);
