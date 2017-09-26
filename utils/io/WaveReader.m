% read waveforms
classdef WaveReader
    properties
        format='wav';            % [binary|raw|wav]
        fs = 16000;              % sampling rate
        precision = 'single';    % [single|int16|double]
        bigEndian = 0;           % [0|1]
        nChannel = 1;            % number of channels. If it is set to 2 or larger, we treat the data as microphone array signals
        channelInMultiFiles = 0; % whether the array channels are in a single file or multiple files: [0|1]
    end
    methods
        function obj = WaveReader(format, fs, precision, bigEndian, nChannel, channelInMultiFiles)
            if nargin>=1; obj.format = format; end
            if nargin>=2; obj.fs = fs; end
            if nargin>=3; obj.precision = precision; end
            if nargin>=4; obj.bigEndian = bigEndian; end
            if nargin>=5; obj.nChannel = nChannel; end
            if nargin>=6; obj.channelInMultiFiles = channelInMultiFiles; end
        end
        function wav = read(obj, files)
            % files is a cell array of file paths if we want to read in 
            % microphone array signals and each channel is stored in an
            % individual file. 
            % otherwise, files is a string containing a file path. 
            % Extra options are allowed in the file paths. 
            % For example: path channel=1,3,5 sample=1,16000 will select
            % the first, third, and fifth channels and samples from 1 to
            % 16000. The terms in the file path should be separated by tab.
            if obj.nChannel>1                   % read in multichannel waveforms
                if obj.channelInMultiFiles      % when the different channels are stored in different files
                    for j=1:length(files)       % files should be a cell array
                        [tmp_wav] = read_a_file(obj,files{j});
                        if j==1; wav = zeros(length(files), length(tmp_wav)); end
                        wav(j,:) = tmp_wav;
                    end
                else                            % when the different channels are stored in the same file
                    [wav] = read_a_file(obj,files);
                end
            else                                % read in single channel waveform
                [wav] = read_a_file(obj,files);
            end
            switch lower(obj.precision)         % put the waveform in correct precision
                case 'int16';   wav = StoreWavInt16(wav);
                case 'single';  wav = single(wav);
                case 'double';  wav = double(wav);
            end
        end
    end
    methods (Access = protected)
        function [wav] = read_a_file(obj, file)
            words = strsplit(file, '\t');
            filename = words{1};
            selectChannel = 0;
            selectSample = 0;
            
            for i=2:length(words)
                terms = strsplit(words{i}, '=');
                if strcmpi(terms{1}, 'channel')
                    selectChannel = 1;
                    channelID = str2num(terms{end});
                elseif strcmpi(terms{1}, 'sample')
                    selectSample = 1;
                    terms2 = strsplit(words{i}, ',');
                    sampleIdx(1) = str2num(terms2{1});
                    sampleIdx(2) = str2num(terms2{2});
                end
            end
            
            switch lower(obj.format)
                case 'raw'
                    [wav] = read08(filename, obj.bigEndian); 
                case 'binary'
                    reader = BinaryReader(obj.nChannel, 'int16', obj.bigEndian);
                    if selectSample
                        [wav] = reader.read(filename, sampleIdx);
                    else
                        [wav] = reader.read(filename);
                    end
                otherwise
                    if selectSample
                        [wav] = audioread(filename, sampleIdx);
                    else
                        [wav] = audioread(filename);
                    end
                    wav = wav';
            end
            if selectChannel
                wav = wav(channelID,:);
            end
        end
    end
end
