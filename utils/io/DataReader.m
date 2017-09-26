% a binary reader class for read and write binary files 

classdef DataReader
    properties
        bigEndian=0;
        vecSize;    % data is treated as matrix, vecSize * nVec
        precision;  % choose from [ubitn|int16]
    end
    methods
        function obj = BinaryReader(vecSize, precision, bigEndian)
            obj.vecSize = vecSize;
            obj.precision = precision;
            if nargin>=3; obj.bigEndian = bigEndian; end
        end
        
        function write(obj, file_name, data)
            FILE = fopen( file_name, 'w' );
            if FILE < 0
                fprintf('File open error: %s\n', file_name); return;
            end
            fwrite(FILE, data(:), obj.precision, GetEndianStr(obj));
            fclose(FILE);
        end
        function [data] = read(obj, file_name, vecRange)
            if nargin<3 || isempty(vecRange)
                readPartial = 0;
            else
                readPartial = 1;
                offset = (vecRange(1)-1) * obj.vecSize * GetSampleSize(obj);
                readSize = (vecRange(2)-vecRange(1)+1) * obj.vecSize;
            end
            
            FILE = fopen( file_name );
            if FILE < 0
                fprintf('File open error: %s\n', file_name);
                data = []; return;
            end
            
            if readPartial
                if strcmpi(obj.precision, 'ubit1')
                    remainderBits = (offset - floor(offset)) * 8;
                    offset = floor(offset);
                    fseek(FILE, offset, 'bof');
                    data = fread(FILE, readSize+remainderBits, obj.precision, GetEndianStr(obj));
                    data(1:remainderBits) = [];
                else
                    fseek(FILE, offset, 'bof');
                    data = fread(FILE, readSize, obj.precision, GetEndianStr(obj));
                end
            else
                data = fread(FILE, obj.precision, GetEndianStr(obj));
            end
            switch obj.precision
                case 'ubit1'    % for 1-bit precision data, we often get extra bits due to that the minimum storage unit is byte. 
                    data = logical(data);
                    nVec = floor(length(data)/obj.vecSize);
                case 'int16'
                    data = int16(data);
                    nVec = length(data)/obj.vecSize;
            end
            data = reshape(data(1:obj.vecSize*nVec), obj.vecSize, nVec);            
            
            fclose(FILE);
        end
    end
    methods (Access = protected)
        function endianStr = GetEndianStr(obj)
            if obj.bigEndian
                endianStr = 'ieee-be';
            else
                endianStr = 'ieee-le';
            end
        end
        function sampleSize = GetSampleSize(obj)
            switch lower(obj.precision)
                case 'ubit1'
                    sampleSize = 0.125;
                case 'int16'
                    sampleSize = 2;
            end
        end
    end
end
