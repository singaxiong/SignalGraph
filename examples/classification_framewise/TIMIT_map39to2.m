% The mapping from 61 phones set to 48 phonse set for TIMIT. The mapping
% table is as follows:
%
% cl vcl epi --> sil
% el --> l
% en --> n
% zh --> sh
% aa --> ao
% ix --> ih
% ax --> ah
%
% Author: Xiong Xiao, NTU, Singapore
% Date: 1 Jun 2016

function phone_seq_2 = TIMIT_map39to2(phone_seq_39)

found = strcmpi(phone_seq_39, 'sil');
phone_seq_2 = phone_seq_39;
idx = find(found==0);
for j=1:length(idx)
    phone_seq_2{idx(j)} = 'speech';
end
end

