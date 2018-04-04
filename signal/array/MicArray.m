classdef MicArray
    properties
        nCh = 2;
        micPosition = [];
        speedOfSound = 340;
        fs = 16000;
        fftSize = 512;
        
        steerVec = [];
    end
    
    methods
        function obj = MicArray(nCh, micPosition)
            obj.nCh = nCh;
            obj.micPosition = micPosition;
        end
        
        % generate steering vector for incoming far field signal, assuming
        % a 2D array and the incoming signal is also at the 2D plane of the
        % array. 
        %
        % Input is the angle of the signal as shown below. "O" is the
        % original of the 
        %             0
        %
        %      270    O-----90
        %
        %            180
        %
        function [steerVec,TDOA] = GenSteeringVec2D(obj, sig_angle)
            if nargin<2; sig_angle = 1:360; end
            
            sig_angle2 = sig_angle/180*pi;
            directionVec = [sin(sig_angle2); cos(sig_angle2); zeros(1,length(sig_angle))];
            
            [steerVec,TDOA] = GenSteeringVecFarField(obj, directionVec);
        end
        
        % generate the steering vector of a far field signal, given a unit
        % norm directional vector of the signal's direction. 
        %
        function [steerVec,TDOA] = GenSteeringVecFarField(obj, directionVec)
            nBeam = size(directionVec,2);
            TDOA = zeros(obj.nCh, nBeam);
            for i=1:nBeam
                tdoa = -obj.micPosition' * directionVec(:,i) / obj.speedOfSound * obj.fs;
                TDOA(:,i) = tdoa;
            end
            
            nBin = obj.fftSize/2+1;
            freq_bin = (0:nBin-1) * pi / (nBin-1);
            steerVec = exp(-sqrt(-1) * bsxfun(@times, freq_bin', permute(TDOA, [3 1 2])) ) / obj.nCh;
        end
        
        function DisplayGeometry2D(obj, newFigure)
            if newFigure; figure; end
            
            xmic = obj.micPosition(1,:);
            arrayWidth = max(xmic) - min(xmic);
            for i=1:obj.nCh
                plot(obj.micPosition(1,i), obj.micPosition(2,i), 'o'); hold on;
                text(obj.micPosition(1,i)+arrayWidth/30, obj.micPosition(2,i), num2str(i));
            end
            hold off;
        end
        
        % generate (normalized) beam patterns for input filters
        function [bp,bpn,bpc] = GetBeampattern(obj, filter)
            filtered = bsxfun(@times, obj.steerVec, permute(conj(filter), [1 2 4 3]));
            bpc = squeeze(sum(filtered,2));
            bp = abs(bpc);
            bpn = bsxfun(@times, bp, 1./max(bp,[],2));
        end
    end
end
