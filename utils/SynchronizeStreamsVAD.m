% Use the VAD stream (the second stream) to select speech frames in the first stream
% Assume the input data is a matrix of DxT size. 
function output = SynchronizeStreamsVAD(data_in, action)
data = data_in{1};
vad = data_in{2};
[D,nFr] = size(data);
precision = class(gather(data_in{1}(1,1,1)));

words = ExtractWordsFromString_v2(action);
action = words{1};

% first
switch lower(action)
    case 'concatenation'
        nFr = size(data,2);
%         [n1,n2,n3] = size(data);
%         if n3>1
%             nFr = n3;
%         else
%             nFr = n2;   % this would be erronous if the input is a tensor with just one frame. But this should be almost impossible as we will not use vad in this case. 
%         end
        vad(nFr+1:end) = [];
%         if n3>1
%             output = data(:,:,vad==1);
%         else
            output = data(:,vad==1);
%         end
        
    case 'segmentation'
        seglen = str2num(words{2});
        segshift = str2num(words{3});
        vad_seg = label2seg(vad);
        output_tmp = {};
        for j=1:length(vad_seg.label)
            if vad_seg.label(j)==0; continue; end
            if vad_seg.stop(j)-vad_seg.start(j) < seglen; continue; end
            start = max(1,vad_seg.start(j)); 
            stop = min(vad_seg.stop(j), nFr);
            output_tmp{end+1} = DivideSent2Segments(data(:,start:stop), seglen, segshift, 0);
        end
        % move data into a tensor
        nSeg = [];
        for j=1:length(output_tmp)
            nSeg(j) = size(output_tmp{j}, 3);
        end
        if length(nSeg) == 0
            pause(0.1);
        end
        output = zeros(D, seglen, sum(nSeg), precision);
        for j=1:length(output_tmp)
            output(:,:, sum(nSeg(1:j-1))+1 : sum(nSeg(1:j)) ) = output_tmp{j};
        end

    otherwise
        fprintf('SynchronizeStreamsVAD: Error: unknown action %s\n', action);
end
    
end
