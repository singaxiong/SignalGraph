function [HEADER_MAT, FEATURE_MAT] = arkread(ark_filename)
%[HEADER_MAT, FEATURE_MAT] = arkread(ark_filename)
%
% Function arkread reads in an ark file (Kaldi feature file) and stores its content in matrices HEADER_MAT and FEATURE_MAT.
% In general, Kaldi ark files may contain feature vectors from multiple token files. 
%
% - HEADER_MAT stores headers of the individual tokens, each row capturing one token.
% The row contains token file name, number of feature frames, number of frame dimensions, start and end byte address 
% of the the raw feature chunk in the ark file, and corresponding start and end row index in the FEATURE_MAT matrix.
%
% - FEATURE_MAT contains concatenated raw feature vectors from all tokens in the ark file. The rows represent individual
% feature frames (and columns their dimensions).
%
% Example:
%
% >>HEADER_MAT
%
% HEADER_MAT = 
%
%   'MDAB0_SI1039'    [392]    [13]    [   28]    [ 20411]    [   1]    [ 392]
%   'MDAB0_SI1669'    [204]    [13]    [20440]    [ 31047]    [ 393]    [ 596]
%
% Features of 'MDAB0_SI1669' can be accessed 
%
% >>HEADER_MAT(393:596, :)

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


HEADER_MAT = [];       % matrix of headers - each row contains token filename, number of frames, number of dimensions per frame, start and end address of the raw feature block, and start and end row index in the FEATURE_MAT matrix where the features will be stored 
FEATURE_MAT = [];      % matrix of concatenated feature vectors from all tokens

FID = fopen(ark_filename, 'r', 'ieee-le');  % 'ieee-le' or 'l' - IEEE floating point with little-endian byte ordering

if (FID == -1)
	err_msg = ['arkread(): Could not open ' ark_filename ', terminating!'];
	disp(err_msg);
	return;
end

file_ascii = fread(FID, 'uchar');

search_pattern = 'BFM ' - 0;  % converts to decimal numbers
indices_BFM = strfind(file_ascii', search_pattern);                     % locate all BFM strings in the ark file

token_start = 0;          % starting index of token's filename
FEATURE_MAT_row_counter = 0;

for k = 1:length(indices_BFM),
	BFM_ind = indices_BFM(k);

	if (fseek(FID, token_start, -1) == -1)
		err_msg = ['arkread(): Seek error! Could not seek to address ' num2str(token_start) ' in ' ark_filename ', terminating!'];
		disp(err_msg);
		break;
	end
	token_fname_len = BFM_ind - 3 - token_start;
	if (token_fname_len < 1)    % some sequence of feature values can accidentally match the ASCII of 'BFM'; if the inequality here is '<', it means a string BFM was found within the chunk of features from previous token, so clearly this 'BFM' is not header-related => skip this BFM_index
		continue;           % this BFM string was in fact features, skipping
	end
	token_fname = char(fread(FID, token_fname_len, 'uchar'))';

	ind_no_frames = BFM_ind + 5 - 1;    % unlike matlab matrices, fseek is indexing from 0
	if (fseek(FID, ind_no_frames, -1) == -1) 
		err_msg = ['arkread(): Seek error! Could not seek to address ' num2str(ind_no_frames) ' in ' ark_filename ', terminating!'];
		disp(err_msg);
		break;
	end
	no_frames = fread(FID, 1, 'uint32');

	ind_no_dimensions = ind_no_frames + 5;
	if (fseek(FID, ind_no_dimensions, -1) == -1) 
		err_msg = ['arkread(): Seek error! Could not seek to address ' num2str(ind_no_dimensions) ' in ' ark_filename ', terminating!'];
		disp(err_msg);
		break;
	end
	no_dimensions = fread(FID, 1, 'uint32');

	ind_fea_start = ind_no_dimensions + 4;
	ind_fea_end = ind_fea_start + no_frames*no_dimensions*4 - 1; 

	if (fseek(FID, ind_fea_start, -1) == -1) 
		err_msg = ['arkread(): Seek error! Could not seek to address ' num2str(ind_fea_start) ' in ' ark_filename ', terminating!'];
		disp(err_msg);
		break;
	end
	FEATURE_MAT_act = vec2mat(fread(FID, (ind_fea_end - ind_fea_start + 1)/4, 'float32'), no_dimensions);
	FEATURE_MAT = [FEATURE_MAT; FEATURE_MAT_act];

	HEADER_MAT = [HEADER_MAT; {token_fname}, no_frames, no_dimensions, ind_fea_start, ind_fea_end, FEATURE_MAT_row_counter + 1, FEATURE_MAT_row_counter + no_frames];

	token_start = ind_fea_end + 1; 
	FEATURE_MAT_row_counter = FEATURE_MAT_row_counter + no_frames;
end
fclose(FID);
