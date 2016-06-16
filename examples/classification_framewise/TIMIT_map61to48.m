% The mapping from 61 phones set to 48 phonse set for TIMIT. The mapping
% table is as follows:
%
% pcl tcl kcl qcl q --> cl
% bcl dcl gcl --> vcl
% h# #h pau --> sil
% ux --> uw
% axr --> er
% em --> m
% nx --> n
% eng --> ng
% hv --> hh
% ax-h --> ax
%
% Author: Xiong Xiao, NTU, Singapore
% Date: 1 Jun 2016

function phone_seq_48 = TIMIT_map61to48(phone_seq_61)

mapping_table = {
    'pcl' 'cl';
    'tcl' 'cl';
    'kcl' 'cl';
    'qcl' 'cl';
    'q'   'cl';
    'bcl' 'vcl';
    'dcl' 'vcl';
    'gcl' 'vcl';
    'h#'  'sil';
    '#h'  'sil';
    'pau' 'sil';
    'ux'  'uw';
    'axr' 'er';
    'em'  'm';
    'nx'  'n';
    'eng' 'ng';
    'hv'  'hh';
    'ax-h' 'ax';};

phone_seq_48 = phone_seq_61;
for i=1:size(mapping_table,1)
    found = strcmpi(phone_seq_48, mapping_table{i,1});
    if sum(found)>0
        idx = find(found==1);
        for j=1:length(idx)
            phone_seq_48{idx(j)} = mapping_table{i,2};
        end
    end
end
end

 
