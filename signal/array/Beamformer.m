classdef Beamformer
    properties
        steerVec = [];
        filter = [];
    end
    methods
        function obj = Beamformer(steerVec)
            if nargin>0; obj.steerVec = steerVec; end
        end
        
        function beamformed = Filter(obj, data, filter)
            if nargin<3
                filter = obj.filter;
            end
            
            beamformed  = bsxfun(@times, data, permute(conj(filter), [1 2 4 3]));
        end
        
        function [bp, bpn] = GetBeamPattern(obj, filter, steerVec)
            if nargin<3
                steerVec = obj.steerVec;
            end
            
            bp = bsxfun(@times, steerVec, permute(conj(filter), [1 2 4 3]));
            bp = squeeze(sum(bp,2));
            bp = abs(bp);

            bpn = bsxfun(@times, bp, 1./max(bp, [], 2));
        end
    end

end
