function [status] = ark2scp(filename_ark)
%[status] = ark2scp(filename_ark)
%
% Function ark2scp reads a Kaldi ark feature file and produces an scp file list in Kaldi format, i.e., each line of the list contains 'utterance' file name followed by the .ark file path and the fast access address.
% To assure that the scp file will contain full path to the ark file, provide full path as an argument to the ark2scp function.

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


filename_scp_out = regexp(filename_ark, '(.*)\.ark', 'tokens');
if (isempty(filename_scp_out))                                   % if the ark file name contains 'ark', replace it by scp. If not, just add .scp behind the whatever ark file name
	filename_scp_out = [filename_ark '.scp'];		
else
	filename_scp_out = [char(filename_scp_out{1}) '.scp'];
end

[HEADER_MAT, FEATURE_MAT] = arkread(filename_ark);
FID_SCP = fopen(filename_scp_out, 'w');
ark_size = size(HEADER_MAT);
number_of_files = ark_size(1);

for z = 1:number_of_files,
	fea_name_raw = char(HEADER_MAT(z, 1));
	address_start = cell2mat(HEADER_MAT(z,4)) - 15;
	fprintf(FID_SCP, '%s %s:%d\n', fea_name_raw, filename_ark, address_start);
end

fclose(FID_SCP);
status = 1;
