function [validFrameMask, variableLength] = getValidFrameMask(input_layer)

if isfield(input_layer, 'validFrameMask')  && ~isempty(input_layer.validFrameMask)  % if there is already the mask, use it
    validFrameMask = input_layer.validFrameMask;
    if nargout==2
        if isempty(validFrameMask)
            variableLength = 0;
        else
            variableLength = sum(validFrameMask(:));
        end
    end
else                                        % otherwise, generate it from the data
    [validFrameMask, variableLength] = CheckTrajectoryLength(input_layer.a);
end

end
