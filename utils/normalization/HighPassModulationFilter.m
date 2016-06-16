function y = HighPassModulationFilter(x, para)

[nFr, dim] = size(x);


switch para.filter_type
    case 'CMN'
        y = CMN(x);
        
    case 'CMN_online'
        y = CMN_online(x);
        
    case 'IIR'
        % Hd = FilterDesignChebchev();
        Hd = FilterDesignButterworth2_1();
        % Hd = FilterDesignCLS0_5();
        
        y = filter(Hd, [x;x;x;x]);
        y= y(end-nFr+1:end,:);
        
    otherwise
        fprintf('Unknown filter type\n');
end


end
