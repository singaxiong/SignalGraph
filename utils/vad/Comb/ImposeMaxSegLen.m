function vad = ImposeMaxSegLen(vad, vad_tight, max_len)
if length(vad)~=length(vad_tight)
    nFr = min(length(vad), length(vad_tight));
    vad = vad(1:nFr); 
    vad_tight = vad_tight(1:nFr);
end

vad_seg = label2seg(vad);
idx = find(vad_seg.label==1);
currDurations = vad_seg.stop(idx)-vad_seg.start(idx);
long_idx = find(currDurations>max_len);

if length(long_idx)==0
    return;
end
start_time = vad_seg.start(idx(long_idx));
stop_time = vad_seg.stop(idx(long_idx));

for k=1:length(start_time)
    sent_len = stop_time(k)-start_time(k);
    local_vad = vad_tight(start_time(k):stop_time(k));
    local_seg = label2seg(local_vad);
    sp_len = [];
    middle_frame = [];
    for m=1:length(local_seg.start)
        if local_seg.label(m)==0
            sp_len(end+1) = local_seg.stop(m) - local_seg.start(m) + 1;
            middle_frame(end+1) = round((local_seg.stop(m) + local_seg.start(m))/2);
        end
    end
    cutting_point = round(choose_cutting_points_by_sp(sent_len, sp_len, middle_frame, max_len));
    vad(cutting_point+start_time(k)-1) = 0;    % add sp to teh VAD flag
end

% verify whether we solved the problem
vad_seg = label2seg(vad);
idx = find(vad_seg.label==1);
currDurations = vad_seg.stop(idx)-vad_seg.start(idx);
long_idx = find(currDurations>max_len);

if length(long_idx)>0
    fprintf('Error: max len is still violated\n');
end
