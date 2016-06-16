function [output, target] = SelectCostFrame(output, target, Cost_layer)

switch Cost_layer.costFrameSelection
    case 'last'     % only comput the cost for the last frame
        if strcmpi(Cost_layer.name, 'cross_entropy') || strcmpi(Cost_layer.name, 'multi_cross_entropy')
            if iscell(output)
                for i=1:length(output)
                    output{i} = output{i}(:,end);
                    target{i} = target{i}(:,end);
                end
            else
                output = output(:,end,:);
                target = target(:,end,:);
            end
        else    % for MSE, just set the previous frames to 0
            if iscell(output)
                for i=1:length(output)
                    output{i}(1:end-1) = 0;
                    target{i}(1:end-1) = 0;
                end
            else
                output(:,1:end-1,:) = 0;
                target(:,1:end-1,:) = 0;
            end
        end
end
end