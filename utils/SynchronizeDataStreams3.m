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
function [data_sync, isVAD] = SynchronizeDataStreams3(alldata, para)
[nStream, nUtt] = size(alldata);

isTensor = para.IO.isTensor;
isVAD = zeros(nStream,1);
data_sync = alldata;

ApplyVADSet = para.IO.ApplyVADSet;
DataSyncSet = para.IO.DataSyncSet;

for vi=1:length(ApplyVADSet)
    if ~isempty(ApplyVADSet{vi})
        isVAD(ApplyVADSet{vi}(end))=1;      % set whether a stream is VAD stream or not
    end
end

if isempty(DataSyncSet) && isempty(ApplyVADSet)
    return;
end

for utt_i = 1:nUtt
    % Apply VAD and segmentation first, if required
    for vi=1:length(ApplyVADSet)
        curr_vad_set = ApplyVADSet{vi};
        if isempty(curr_vad_set); continue; end
        data_sync{curr_vad_set(1), utt_i} = SynchronizeStreamsVAD( data_sync(curr_vad_set, utt_i), para.IO.vadAction{vi} );
    end
    
    % Apply Frame rate synchronization to all Sync Sets
    for di=1:length(DataSyncSet)
        curr_sync_set = DataSyncSet{di};
        if isempty(curr_sync_set); continue; end
        data_sync(curr_sync_set, utt_i) = SynchronizeStreamsFrameRate2(  data_sync(curr_sync_set, utt_i) , para.IO.frame_rate(curr_sync_set) , isTensor(curr_sync_set)  );
    end    
end
end