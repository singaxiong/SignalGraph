classdef TextGridIO
    
    methods
        function [time, label, tierName] = ReadTextGrid(obj, inputFile)
            input = my_cat(inputFile);
            
            time = {};
            label = {};
            tierName = {};
            
            index = 1;
            while index<length(input)
                [time{end+1}, label{end+1}, tierName{end+1}, index] = obj.ReadTextGridTier(input, index);
            end
            
        end
        
        function WriteTextGridFromTSV(obj, fileName, text)
            nSeg = length(text);
            text2 = {};
            for i=1:length(text)
                terms = strsplit(text{i}, '\t');
                times = str2num(terms{1});
                text2{end+1} = num2str(times(1));
                text2{end+1} = num2str(times(2));
                text2{end+1} = ['"' terms{3} '"'];
            end
            endTime = times(2);
            
            header{1} = 'File type = "ooTextFile"';
            header{end+1} = 'Object class = "TextGrid"';
            header{end+1} = '';
            header{end+1} = '0';
            header{end+1} = num2str(endTime);
            header{end+1} = '<exists>';
            header{end+1} = '1';
            header{end+1} = '"IntervalTier"';
            header{end+1} = sprintf('"%s"', 'abc');
            header{end+1} = '0';
            header{end+1} = num2str(endTime);
            header{end+1} = num2str(nSeg);
            
            text2 = [header text2];
            my_dump(fileName, text2);
        end
    end
    
    methods (Access = protected)
        function [time, label, TierName, pointer] = ReadTextGridTier(obj, input, index)
            
            for i=index:length(input)
                if strcmpi(input{i}, '"IntervalTier"'); break; end
            end
            TierName = input{i+1}(2:end-1);
            start_idx = i+2;
            
            nSeg = ceil((length(input)-start_idx) / 3);   % rough estimation of the number of segments
            time = zeros(2, nSeg);
            segCnt = 0;
            
            i=start_idx;
            existNextTier = 0;
            while i<=length(input)
                if strcmpi(input{i}, '"IntervalTier"')
                    existNextTier = 1;
                    break;
                end
                if input{i}(1) == '"'
                    segCnt = segCnt + 1;
                    time(1, segCnt) = str2num(input{i-2});
                    time(2, segCnt) = str2num(input{i-1});
                    label{segCnt} = input{i}(2:end-1);
                end
                i = i+1;
            end
            time(:,segCnt+1:end) = [];
            pointer = i;
        end
    end
    
end
