% This class contains the data of one input stream of signal graph. 
% It should either contain the data itself, or file name of the data. 
%   if use file names, should also have the file reader for the files. 
% It also contains the necessary information on how to use the data. 

classdef DataStream
    properties
        % Configurations
        baseType = 'matrix'; %  --- the format of the base: 1) matrix; 2) tensor; 3) cell. Default value is matrix.
        isFile = 0; % whether the data is file names
        isIndex = ''; % whether the data is index
        isSparse = 0; % whether the data is sparse
        frameRate = 100; %--- the number of frames per second. Default value is 100.
        fileReader = []; % [WaveReader|]

        % data
        data = [];  % the actuall data of the stream
    end
    methods
        function obj = DataStream()
            
        end
        
        function data = getData(obj,idx,datatype)
            if nargin<3; datatype = 'single'; end
            if nargin<2 || isempty(idx)
                idx = 1:length(obj.data);
            end
            nUtt = length(idx);
            for i=1:nUtt
                PrintProgress(i, nUtt, max(50,ceil(nUtt/10)), 'DataStream: load stream data');
                if obj.isFile
                    data{i} = obj.read(obj.data{idx(i)});
                else
                    data{i} = obj.data{idx(i)};
                end
                if strcmpi(datatype, 'single')
                    data{i} = single(data{i});
                else
                    data{i} = double(data{i});
                end
                data{i} = full(data{i});    % sometimes, the data is in sparse format. We need to convert them to full

                % sometimes, we want to generate large amount of data from
                % a fixed set of basis data. This part of code has not been
                % adapted from the old code yet. 
%                 if para.IO.isIndex(i)   % if the input is an index, retrieve the real data from the base.
%                     tmp_feat_idx = tmp_feat;
%                     switch para.IO.baseType
%                         case 'matrix'
%                             tmp_feat = data(i).base(:,tmp_feat_idx);
%                         case 'tensor'
%                             tmp_feat = data(i).base(:,:,tmp_feat_idx);
%                         case 'cell'
%                             tmp_feat = data(i).base(tmp_feat_idx);
%                     end
%                 end

            end
        end
        function apprxSampleSize = getApprxSampleSize(obj)
            if obj.isFile
                % randomly read several files and compute the average
                idx = randperm(length(obj.data), 10);
                for i=1:length(idx)
                    tmpData{i} = obj.read(obj.data{idx(i)});
                end
                allSize = cell2mat(cellfun(@size, tmpData', 'UniformOutput', 0));
                sampleSize = allSize(:,1) .* allSize(:,2);
                apprxSampleSize = mean(sampleSize);
            else
                allSize = cell2mat(cellfun(@size, obj.data, 'UniformOutput', 0));
                sampleSize = allSize(:,1) .* allSize(:,2);
                apprxSampleSize = mean(sampleSize);
            end
        end
    end
    methods (Access = protected)
        function currSample = read(obj, fileName)
            currSample = obj.fileReader.read(fileName);
        end
    end
end
