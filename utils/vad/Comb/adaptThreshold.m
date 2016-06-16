%nv and var are disable.
%they are not used in computation of VAD
%
function [s_new] = adaptThreshold(s_old, val)

if (s_old.frameIdx <= 1)
  s_old.val = val;
end

diffVal      = abs(val-s_old.val);
s_new     = s_old;

s_new.frameIdx = s_new.frameIdx+1;


if (s_new.frameIdx < 3)
    s_new.vadFlag  = 0;
    s_new.countVADON = 0;
    if (s_new.frameIdx == 1)
        s_new.nf = s_new.val;
        s_new.slow_nf = s_new.val;
    end
    s_new.nf          = max(s_new.nf, s_new.val);
    s_new.slow_nf     = max(s_new.slow_nf, s_new.val);
    s_new.speechVar  =  s_new.MinRatioSpeechPeakNoiseFloor*s_new.nf + s_new.ThOffset;
    s_new.th           =  s_new.slow_nf     + s_new.ThOffset;
    s_new.th2          =  s_new.slow_nf*1.3 + s_new.ThOffset;
end

s_new.val = val;

if (s_new.frameIdx >= 3)

  s_new.speechVar = 0.8*s_new.speechVar + 0.2*s_new.val;
  if (s_new.speechVar < s_new.th2)
      s_new.speechVar = s_new.th2;
  end
      
      
  if (s_new.val > s_new.th2)
        s_new.vadFlag = 1;       
        if (s_new.val > s_new.speechVar)
            s_new.speechVar = s_new.val;
        end
  else
       s_new.vadFlag = 0;        
  end

  s_new.nf = 0.99*s_new.nf + 0.01*s_new.val;    
  s_new.slow_nf = 0.999*s_new.slow_nf + 0.001*s_new.val;    
  if (s_new.val < s_new.slow_nf)
     s_new.slow_nf = s_new.val; 
     if (s_new.slow_nf < s_new.MinFeatureVal)
         s_new.slow_nf  = s_new.MinFeatureVal;
     end
  end
  
  if (s_new.val < s_new.nf)
        s_new.nf = s_new.val;

        if (s_new.nf < s_new.MinFeatureVal)
            s_new.nf = s_new.MinFeatureVal;
        end
  end

  
   s_new.th         =   s_new.nf     + s_new.ThOffset;
   s_new.th2        =   s_new.nf*1.3 + s_new.ThOffset;
 
end


