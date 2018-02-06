classdef WaveBuffer
    properties
        nCh = 1;            % number of channels
        BufferSize = 400;   % this is the buffer size, and also the data size we get from the buffer everytime we call Get().
        BlockSize = 160;    % this is the amount of data we push into the buffer everytime we call Push.
        FrameOverlap = 240; % this is the overlap between two consecutive frames.
        % Note that, the BlockSize = BufferSize - FrameOverlap. In this way,
        % we can gauarantee that the blockwise processing generate the same
        % Short-time Fourier Transform representation of the signal as in
        % the offline processing.
        precision = 'double';
        data = [];
    end
    
    methods
        function obj = WaveBuffer(nCh, BufferSize, FrameOverlap, precision)
            obj.nCh = nCh;
            obj.BufferSize = BufferSize;
            obj.FrameOverlap = FrameOverlap;
            obj.BlockSize = BufferSize - FrameOverlap;
            if nargin>=4
                obj.precision = precision;
            end
            
            % initialize the buffer with 0's
            obj.data = zeros(obj.nCh, obj.BufferSize, obj.precision);
        end
        
        function obj = Push(obj, data)  % push data into the buffer
            [nCh, nSample] = size(data);
            
            assert(obj.nCh == nCh, sprintf('Error: number of channel in data (%d) is different from number of channel in buffer (%d)', nCh, obj.nCh));
            assert(obj.BlockSize == nSample, sprintf('Error: number of samples in data (%d) is different from block size(%d)', nSample, obj.BlockSize));
            
            obj.data = [obj.data(:,obj.BlockSize+1:end) data];
        end
        
        function data = Get(obj)        % get the data from the buffer
            data = obj.data;
        end
        
        function obj = Reset(obj)       % reset the content of the buffer to 0
            obj.data(:) = 0;
        end
    end
    
end
