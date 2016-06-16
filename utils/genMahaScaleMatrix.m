function [scale_matrix, phoneID] = genMahaScaleMatrix(phones, alignment, std_recp, mode)

if nargin<4
    mode = 1;
end

if mode==1
    [phoneID] = alignment2phoneID(phones, alignment);
    phoneID = cell2mat(phoneID);
    for i=1:length(phones)
        idx = phoneID==i;
        scale_matrix(:,idx) = repmat(std_recp(:,i), 1, sum(idx));
    end
elseif mode==2
    nPhone = length(phones);
    for i=1:nPhone
        idx = alignment==phones(i);
        scale_matrix(:,idx) = repmat(std_recp(:,i), 1, sum(idx));
    end    
end