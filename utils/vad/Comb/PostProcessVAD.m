function [final_vad, vad_extended] = PostProcessVAD(vad, buffer_len, filter_len)
vad = vad(:);

% First filter the vad flat using a median filter
vad_smoothed = MedianFilter(vad, 3);
% Then grow any valid speech frame by 20 frames, as we think
% anything immediately before and after the voiced frames are
% speech.
if nargin<3
    filter_len = 40;
end
vad_extended = double(conv(ones(filter_len,1), vad_smoothed)>0.5);
vad_extended(1:filter_len/2) = [];
vad_extended(end-filter_len/2+2:end) = [];

% We don't want to have too many segments. Sometimes, we can merge
% several segments into one.
filter_len = buffer_len-40;
if filter_len>0
    vad_merged = double(conv(ones(filter_len,1), vad_extended)>0.5);
    vad_merged(1:filter_len/2) = [];
    vad_merged(end-filter_len/2+2:end) = [];
else
    vad_merged = vad_extended;
end
    
final_vad = vad_merged;
