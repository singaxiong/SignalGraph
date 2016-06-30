function [output, target] = SelectCostFrame(output, target, Cost_layer)

words = ExtractWordsFromString_v2(Cost_layer.costFrameSelection);
selectionType = words{1};

switch selectionType
    case 'last'     % only comput the cost for the last frame
        if length(words)>1
            N = str2num(words{2});
        else
            N = 1;
        end
        if strcmpi(Cost_layer.name, 'cross_entropy') || strcmpi(Cost_layer.name, 'multi_cross_entropy')
            if iscell(output)
                for i=1:length(output)
                    output{i} = output{i}(:,max(1,end-N+1):end);
                    target{i} = target{i}(:,max(1,end-N+1):end);
                end
            else
                output = output(:,max(1,end-N+1):end,:);
                target = target(:,max(1,end-N+1):end,:);
            end
        else    % for MSE, just set the previous frames to 0
            if iscell(output)
                for i=1:length(output)
                    output{i}(1:end-N) = 0;
                    target{i}(1:end-N) = 0;
                end
            else
                output(:,1:end-N,:) = 0;
                target(:,1:end-N,:) = 0;
            end
        end
    case 'first'
        % to be implemented
end
end