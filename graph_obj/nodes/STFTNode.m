classdef STFTNode < GraphNode
    properties
        fft_len=512;
        frame_len=400;
        frame_shift=160;
        win_type = 'hamming';
        removeDC = 1;
        doDithering = 1;
    end
    
    methods
        function obj = STFTNode(dimOut)
            obj = obj@GraphNode('STFT',dimOut);
        end
        
        function obj = forward(obj,prev_layers)
            input = prev_layers{1}.a;
            [nCh,D2,T,N] = size(input);
            assert(D2==1, sprintf('%s:forward, Error: dimension 2 (%d) is not equal to 1', obj.name, D2));
            input = permute(input, [1 3 4 2]);
            
            precision = class(gather(input(1)));
            useGPU = IsInGPU(input);
            
            if N==1
                fft_x = sfft_multi(input',obj.frame_len,obj.frame_shift,obj.fft_len, obj.win_type,obj.removeDC, useGPU, obj.doDithering);
                fft_x = fft_x(1:obj.fft_len/2+1,:,:);
                [d1,d2,d3] = size(fft_x);
                fft_x = reshape(fft_x, d1*d2, d3);
                obj.mask = [];
            else
                [mask, variableLength] = CheckTrajectoryLength(input);
                input2 = PadShortTrajectory(input, mask, 0);
                input2 = reshape(permute(input2, [2 1 3]), T, nCh*N);
                
                fft_x = sfft_multi(input2,obj.frame_len,obj.frame_shift,obj.fft_len, obj.win_type,obj.removeDC, useGPU, obj.doDithering);
                fft_x = fft_x(1:obj.fft_len/2+1,:,:);
                fft_x = permute(fft_x, [1 3 2]);
                [nBin,nFr,d3] = size(fft_x);
                fft_x = reshape(fft_x, nBin, nFr, nCh, N);
                fft_x = permute(fft_x, [1 3 2 4]);
                fft_x = reshape(fft_x, nBin*nCh, nFr, N);
                
                % now build a mask for spectrogram
                nSampleChannel = sum(mask==0);
                nSampleChannel = gather(nSampleChannel);
                if useGPU
                    obj.mask = gpuArray.zeros(nFr, N, precision);
                else
                    obj.mask = zeros(nFr, N, precision);
                end
                for i=1:N
                    nFrChannel = enframe_decide_frame_number(nSampleChannel(i), obj.frame_len, obj.frame_shift, 0);
                    obj.mask(nFrChannel+1:end,i) = 1;
                end
            end
            obj.a = permute(fft_x, [1 4 2 3]);
            obj = forward@GraphNode(obj, prev_layers);
        end
        
        function obj = backward(obj,future_layers, prev_layers)
            % to be implemented
        end
        
    end
    
end