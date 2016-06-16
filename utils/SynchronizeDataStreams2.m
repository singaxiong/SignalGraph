% Synchronoise the streams such that they will work with the network.
% Things did in this function:
%   1. upsample streams to match other streams of higher frame rate. This
%   is only done when streams are specified by DataSyncSet to have the same
%   frame rate. Also discarded last frames longer streams to make the
%   streams have the same number of frames.
%   2. apply VAD to some streams. Here VAD is stored in one stream
%   (normally the last stream). We first upsample/downsample VAD framerate
%   to match the stream to be VADed. Then remove nonspeech segments. VAD
%   should be stored as 1-D feature stream.
%
% Author: Xiong Xiao, Temasek labs, NTU, Singapore.
% Date Created: 10 Oct 2015
% Last Modified: 21 Apr 2016
%
function [data_sync, nFr_actual, isTensor, isVAD, isFramewise] = SynchronizeDataStreams2(alldata, para)
[nStream, nUtt] = size(alldata);

isTensor = para.IO.isTensor;
isVAD = zeros(nStream,1);
isFramewise = zeros(nStream,1);
data_sync = alldata;
nFr_actual = zeros(nUtt, nStream);

ApplyVADSet = para.IO.ApplyVADSet;
DataSyncSet = para.IO.DataSyncSet;
if isempty(DataSyncSet) && isempty(ApplyVADSet)
    for utt_i = 1:nUtt
        for i=1:nStream
            [~,n2,n3] = size(data_sync{i, utt_i});
            if isTensor(i)
                if n2==0
                    nFr_actual(utt_i,i) = 0;
                else
                    nFr_actual(utt_i,i) = n3;
                end
            else
                nFr_actual(utt_i,i) = n2;
            end
            if n2>1
                isFramewise(i) = 1;     % if any sentence has more than one frame, it is framewise feature or label
            end
        end
    end
    return;
end

for utt_i = 1:nUtt
    % Apply VAD and segmentation first, if required
    for vi=1:length(ApplyVADSet)
        isVAD(ApplyVADSet{vi}(end))=1;
        curr_vad_set = ApplyVADSet{vi};
        if isempty(curr_vad_set); continue; end
        data_sync{curr_vad_set(1), utt_i} = SynchronizeStreamsVAD( data_sync(curr_vad_set, utt_i), para.IO.vadAction{vi} );
    end
    
    % Apply Frame rate synchronization to all Sync Sets
    for di=1:length(DataSyncSet)
        curr_sync_set = DataSyncSet{di};
        if isempty(curr_sync_set); continue; end
        data_sync(curr_sync_set, utt_i) = SynchronizeStreamsFrameRate(  data_sync(curr_sync_set, utt_i) , para.IO.frame_rate(curr_sync_set) , isTensor(curr_sync_set)  );
    end    
    
%     for i=1:nStream
%         [~,~,n3] = size(data_sync{i, utt_i});
%         if n3>1; isTensor(i) = 1;   end
%     end
end
for utt_i = 1:nUtt
    for i=1:nStream
        [~,n2,n3] = size(data_sync{i, utt_i});
        if isTensor(i)
            if n2==0
                nFr_actual(utt_i,i) = 0;
            else
                nFr_actual(utt_i,i) = n3;
            end
        else
            nFr_actual(utt_i,i) = n2;
        end
        if n2>1
            isFramewise(i) = 1;     % if any sentence has more than one frame, it is framewise feature or label
        end
    end
end
end