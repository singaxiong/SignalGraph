% This function recursively cuts a sentence into shorter segments no longer 
% than a predefined maximum length, based purely on the information about
% short pause durations and locations. 
% Inputs:
%   sent_len: length of the sentence in terms of frames
%   sp_len: length of the short pauses in the sentence
%   middle_frame: middle of the short pauses
%   max_seg_len: maximum length of every resulting segments. 
%   weight_location: the weight of short pause location. This is relative to the
%       short pause duration
% Outputs:
%   cutting_point: an array of suggested cutting points
% 
% Authors: Xiong Xiao 
% Date Created: 2009
% Last Modified: 16 April 2014
%
function cutting_point = choose_cutting_points_by_sp(sent_len, sp_len, middle_frame, max_seg_len, loc_weight)
if nargin<5
    loc_weight = 1;
end
if sent_len < max_seg_len
    cutting_point = [];
elseif length(sp_len)<2
    cutting_point = [];
else
    % Choose the sp for splitting
    % Two rules to select a sp: 1) longer sp segments should be
    % given higher priority; 2) sp near to center of the sentence
    % should be given higher priority. The weight of these two
    % criteria will be adjsuted by a weighting loc_weight
    sent_center = round(sent_len/2);
    score = [];
    for k = 1:length(sp_len)
        if middle_frame(k) < max_seg_len/5 || sent_len-middle_frame(k)<max_seg_len/5    % we don't cut by short pause that is too close to the both ends
            score(k) = 0;
        else
            dist_to_center = abs(middle_frame(k)-sent_center)/sent_center;      % how far is current short pause to the sentence center relatively, which is in [0 0.5] interval. 
            % we change the distance to log scale, multiple it by -3 to convert it to positive number. As log(0) is infinity, we need to set a upper limit to the log distance. 
            location_score = min(20, -3*log(dist_to_center) );              
            score(k) = sp_len(k) + loc_weight * location_score;     % we still have loc_weight to tune the relative weight of the two factors
        end
    end
    [best_sp, idx] = max(score);    % everytime we only cut the current sentence once. 
    cutting_point = middle_frame(idx);
    
    sub_points = choose_cutting_points_by_sp(cutting_point, sp_len(1:idx-1), middle_frame(1:idx-1), max_seg_len, loc_weight);     % See whether left segment needs to be cut further
    cutting_point = ([sub_points cutting_point]);
    sub_points = choose_cutting_points_by_sp(sent_len - cutting_point(end), sp_len(idx+1:end), middle_frame(idx+1:end)-cutting_point(end), max_seg_len, loc_weight);     % See whether left segment needs to be cut further
    cutting_point = ([cutting_point sub_points+cutting_point(end)]);
end
