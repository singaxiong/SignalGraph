% This function is used for loading speech data of various form, such as
% wav, filterbank, HTK or Kaldi format features, etc.
% Inputs:
%   file_list: list of files to be read. Should contain absolute path.
%   para.data_type: type of data. Currently support:
%       raw_wav: i.e. no header wav stored in 2-byte precision. Need to
%           define whether to use big endian or little endian.
%       sphere_wav: will simply discard the first 1024 bytes. Need to defin
%           whether to use big endian or little endian.
%       htk: htk feature format.
%       kaldi: kaldi feature archive format.
%   processing: processing to be applied on each utterance.
% Author: Xiong Xiao, Temasek Lab, NTU, Singapore. 
% Created: 19 Nov 2013. 
%
function feat = UniversalDataLoader(file_list, para, processing)
msgPrefix = 'IO/UniversalDataLoader';
fprintf('%s Loading data...\n', msgPrefix);
if isfield(para, 'big_endian')
    big_endian = para.big_endian;
else
    big_endian = 1;
end

step = max(1, floor(length(file_list)/10));

feat = {};
for i=1:length(file_list)
    PrintProgress(i, length(file_list), step);
    
    % Read files
    switch lower(para.data_type)
        case 'raw_wav'
            tmp = read08(file_list{i}, big_endian);
        case 'sphere_wav'
            tmp = readNIST(file_list{i}, big_endian);
        case 'htk'
            tmp = readHTK(file_list{i}, [], big_endian)';
        case 'kaldi'
            [~, tmp] = readKaldiFeature(file_list{i});
        otherwise
            fprintf('%s Error: unknown data type %s!\n', msgPrefix, para.data_type);
    end
    
    % Apply some processing
    if length(processing)>0
        if strcmpi(para.data_type, 'kaldi')
            for j=1:length(tmp)
                feat{end+1} = FeaturePipe(tmp{j}, processing);
            end
        else
            feat{end+1} = FeaturePipe(tmp{j}, processing);
        end
    else
        if strcmpi(para.data_type, 'kaldi')
            feat = [feat tmp];
        else
            feat{i} = tmp;
        end
    end
end
end
