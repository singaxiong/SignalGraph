classdef SpectrumAnalyzer
    properties
        fs = 16000;
        nCh = 1;
        nFFT = 512;
        nBin = 257;
        frameSize = 400;
        frameShift = 160;
        frameOverlap = 240;
        frameRate = 100;
        window = 'hamming';
        doDithering = 1;
        doDCRemoval = 0;
        useGPU = 0;
    end
    
    methods
        function obj = SpectrumAnalyzer(fs, nCh)
            obj.fs = fs;
            obj.nCh = nCh;
        end
        
        function spec = Analyze(obj, data)
            spec = sfft_multi(data, obj.frameSize, obj.frameShift, obj.nFFT, obj.window, obj.doDCRemoval, obj.useGPU, obj.doDithering);
            spec = spec(1:obj.nBin,:,:);
        end
        
        function wav = Synthesize(obj, complexSpectrum)
            [nBin,D2,D3] = size(complexSpectrum);
            if D2>1 && D3>1     % if both D2 and D3 > 1, assume D2 is the number of channels, while D3 is the number of frames (same format as sfft_multi). 
                complexSpectrum = permute(complexSpectrum, [1 3 2]);
            end
            for i=1:size(complexSpectrum,3)
                data = squeeze(complexSpectrum(:,:,i));
                data(obj.nFFT/2+2:obj.nFFT,:) = conj(data(obj.nFFT/2:-1:2,:));
                xi = real(ifft(data));
                wav(:,i) = obj.ola(xi);
            end
        end
        
        function DisplaySpectrogram(obj,data, newFigure)
            if nargin<3; newFigure = 1; end
            if newFigure; figure; end
            [nBin, nCh, nFr] = size(data);
            imagesc(log(abs(reshape(data, nBin*nCh, nFr))));
        end
    end
    
    methods (Access = protected)
        function xfinal = ola(obj, xi)
            nFr = size(xi,2);
            k=1;
            xfinal=zeros(nFr*obj.frameShift,1);
            x_old = zeros(obj.frameOverlap,1);
            for j=1:nFr
                xfinal(k:k+obj.frameOverlap-1) = x_old + xi(1:obj.frameOverlap,j);
                x_old = xi(1+obj.frameShift:obj.frameSize,j);
                k = k + obj.frameShift;
            end
        end
    end
end
