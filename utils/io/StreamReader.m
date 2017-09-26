
classdef StreamReader
    properties
        reader;
        streamType;
        precision = 'single';
        bigEndian = 0;
    end
    methods
        function obj = StreamReader(reader, streamType, precision, bigEndian)
            obj.reader = reader;
            obj.streamType = streamType;
            if nargin>=3; obj.precision = precision; end
            if nargin>=4; obj.bigEndian = bigEndian; end
        end
        function data = read(obj, files)
            switch lower(obj.streamType)
                case 'binary'
                    [data] = Reader_binary(files, obj.reader);
                case {'wavfile','wav', 'raw'}
                    [data] = Reader_waveform(files, obj.reader);
                case 'htk'
                    data = Reader_HTK(files, obj.big_endian, obj.precision);
                otherwise
                    fprintf('Unkonwn reader type\n');
            end
            
            switch lower(obj.precision)
                case 'int16'
                    data = StoreWavInt16(data);
                case 'single'
                    data = single(data);
                case 'double'
                    data = double(data);
            end
        end
    end
    methods (Access = protected)        
        function [wav, fs] = Reader_waveform(files, reader)
            if isfield(reader, 'array') && reader.array
                % read in multichannel waveforms
                if isfield(reader, 'multiArrayFiles') && reader.multiArrayFiles     % when the different channels are stored in different files
                    for j=1:length(files)
                        [tmp_wav, fs] = Reader_waveform_core(files{j}, reader.fs);
                        if j==1
                            wav = zeros(length(files), length(tmp_wav));
                        end
                        wav(j,:) = tmp_wav;
                    end
                else                                                                % when the different channels are stored in the same file
                    [wav, fs] = Reader_waveform_core(files, reader.fs);
                end
                % optional selection of channels
                if isfield(reader, 'useChannel')
                    wav = wav(reader.useChannel, :);
                end
            else
                % read in single channel waveform
                [wav, fs] = Reader_waveform_core(files, reader);
            end
        end
        
        function [wav, fs] = Reader_waveform_core(file, reader)
            words = ExtractWordsFromString_v2(file);
            selectChannel = 0;
            selectTime = 0;
            
            filename = words{1};
            if length(words)==4 || length(words)==2
                channelID = str2num(words{end});
                selectChannel = 1;
            end
            if length(words)>=3
                time1 = str2num(words{2});
                time2 = str2num(words{3});
                selectTime = 1;
            end
            
            fs = ReturnFieldWithDefaultValue(reader, 'fs', -1);
            big_endian = ReturnFieldWithDefaultValue(reader, 'big_endian', 0);
            
            switch lower(reader.name)
                case 'raw'
                    [wav] = read08(filename, big_endian);   % you have to set fs correctly in the reader.
                case 'binary'
                    if selectTime
                        [data] = reader.reader.read(filename, [time1 time2]);
                    else
                        [data] = reader.reader.read(filename);
                    end
                otherwise
                    if selectTime
                        [wav, fs] = ReadAudioSeg(filename, time1, time2, fs);
                    else
                        [wav, fs] = audioread(filename);
                    end
            end
            
            if selectChannel
                wav = wav(:,channelID);
            end
            
            wav = wav';
        end
    end
end
