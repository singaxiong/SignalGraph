function [target phones] = alignment2phonePosterior(alignment, phones, mode)
if nargin < 3
    mode = 1;
end
if length(phones) == 0
    phones = {};
    for i=1:length(alignment)
        phones = [phones alignment{i}.label];
        
        if mod(i,100)==0
            phones = unique(phones);
        end
    end
    phones = unique(phones);
end
nPhone = length(phones);

N = 0;
for j=1:length(alignment)
    N = N + alignment{j}.stop(end);
end
if mode == 1    % output posterior matrix
    target = zeros(nPhone, N);
    for i=1:nPhone
        offset = 0;
        for j=1:length(alignment)
            for k=1:length(alignment{j}.label)
                if strcmp(phones{i}, alignment{j}.label{k})
                    target(i, alignment{j}.start(k)+offset : alignment{j}.stop(k)+offset) = 1;
                end
            end
            offset = offset + alignment{j}.stop(end);
        end
    end
elseif mode == 2 % output posterior vector
    target = zeros(1, N);
    for i=1:nPhone
        offset = 0;
        for j=1:length(alignment)
            match = strcmp(alignment{j}.label, phones{i});
            matchIdx = find(match==1);
            for k=matchIdx
                target(1, alignment{j}.start(k)+offset : alignment{j}.stop(k)+offset) = i;
            end
            offset = offset + alignment{j}.stop(end);
        end
    end
elseif mode == 3 % output posterior cell array, one cell per utterance
    for j=1:length(alignment)
        for i=1:nPhone
            match = strcmp(alignment{j}.label, phones{i});
            matchIdx = find(match==1);
            for k=matchIdx
                target{j}(1, alignment{j}.start(k) : alignment{j}.stop(k)) = i;
            end
        end
    end
end
end