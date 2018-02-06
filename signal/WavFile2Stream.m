% Giving a wave file, simulate the audio stream

classdef WavFile2Stream
    properties
        nCh = 1;            % number of channels
        fs = 16000;         % sampling rate
        BufferSize = 400;   % this is the buffer size, and also the data size we get from the buffer everytime we call Get().
        BlockSize = 160;    % this is the amount of data we push into the buffer everytime we call Push.
        FrameOverlap = 240; % this is the overlap between two consecutive frames.
        % Note that, the BlockSize = BufferSize - FrameOverlap. In this way,
        % we can gauarantee that the blockwise processing generate the same
        % Short-time Fourier Transform representation of the signal as in
        % the offline processing.
        precision = 'double';
        
        CurrDataIdx = 1;    % the index of the first sample of next block
        blockData = [];
        
    end
    
    properties (Access = protected)
        data = [];
        buffer;
        nSample = 0;
    end
    
    methods
        function obj = WavFile2Stream(BlockSize, FrameOverlap, precision, wavfile, dataIdx)
            obj.BlockSize = BlockSize;
            obj.FrameOverlap = FrameOverlap;
            obj.BufferSize = BlockSize + FrameOverlap;
            obj.precision = precision;
            obj.CurrDataIdx = dataIdx;
            
            if ischar(wavfile)
                [obj.data, obj.fs] = audioread(wavfile);
                obj.data = obj.data';
            else
                obj.data = wavfile;
            end
            [obj.nCh, obj.nSample] = size(obj.data);
            obj.buffer = WaveBuffer(obj.nCh, obj.BufferSize, obj.FrameOverlap, obj.precision);
        end
        
        
        function obj = GetNextBlock(obj, dataIdx)  % push data into the buffer
            if nargin<2
                dataIdx = obj.CurrDataIdx;  % if no index is given, just use the stored index
            else        % if we are jumping to another data index, we should reset the buffer
                if obj.CurrDataIdx ~= dataIdx
                    obj.buffer = obj.buffer.Reset();
                end
            end
            
            blkEndIdx = dataIdx+obj.BlockSize-1;
            if blkEndIdx > obj.nSample
                obj.blockData = [];
                return;
            end
            
            nextBlock = obj.data(:, dataIdx:blkEndIdx );
            % push the next block of data into the buffer
            obj.buffer = obj.buffer.Push(nextBlock);
            % and then get the whole buffer out.
            obj.blockData = obj.buffer.Get();
            obj.CurrDataIdx = blkEndIdx+1;
        end
        
    end
end
