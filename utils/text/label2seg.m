function seg = label2seg(label)

% If the state is changed
diff = label(2:end) - label(1:end-1);
idx = find(diff ~= 0);
N_seg = length(idx);
if N_seg ==0
    seg.start(1) = 1;
    seg.stop(1) = length(label);
    seg.label(1) = label(1);
    return;
end

for i=1:N_seg
    if i==1
        seg.start(i) = 1;
    else
        seg.start(i) = idx(i-1)+1;
    end
    seg.stop(i) = idx(i);
    seg.label(i) = label(idx(i));
end
seg.start(end+1) = idx(end)+1;
seg.stop(end+1) = length(label);
seg.label(end+1) = label(end);
    