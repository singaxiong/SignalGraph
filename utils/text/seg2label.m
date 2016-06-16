function label = seg2label(seg)

if isfield(seg, 'label')
    N_seg = length(seg.label);
    for i=1:N_seg
        label( seg.start(i) : seg.stop(i) ) = seg.label(i);
    end
elseif isfield(seg, 'ID')
    N_seg = length(seg.ID);
    for i=1:N_seg
        label( seg.time1(i) : seg.time2(i) ) = seg.ID(i);
    end
end