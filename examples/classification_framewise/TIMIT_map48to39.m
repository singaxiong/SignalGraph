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

function phone_seq_39 = TIMIT_map48to39(phone_seq_48)

mapping_table = {
    'cl'  'sil';
    'vcl' 'sil';
    'epi' 'sil';
    'el'  'l';
    'en'  'n';
    'zh'  'sh';
    'aa'  'ao';
    'ix'  'ih';
    'ax'  'ah';};

phone_seq_39 = phone_seq_48;
for i=1:size(mapping_table,1)
    found = strcmpi(phone_seq_39, mapping_table{i,1});
    if sum(found)>0
        idx = find(found==1);
        for j=1:length(idx)
            phone_seq_39{idx(j)} = mapping_table{i,2};
        end
    end
end
end

 
