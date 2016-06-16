% this function compute the derivatives of the static coefficients. Its
% implementation follows that of the HMM Toolkit 3.2

function delta_coef = comp_delta(static_coef, DELTAWINDOW)

[N_vec, N_cep] = size(static_coef);

if 0
    first_vec = static_coef(1,:)';
    last_vec = static_coef(N_vec,:)';
    static_coef = static_coef';
    for i = 1:DELTAWINDOW
        static_coef = [first_vec static_coef];    % append the first feature vector DELTAWINDOW times in the front
        static_coef = [static_coef last_vec];    % append the last feature vector DELTAWINDOW times in the back
    end
    static_coef = static_coef';
else
    idx = [ones(1,DELTAWINDOW) 1:N_vec ones(1,DELTAWINDOW)*N_vec];
    static_coef = static_coef(idx,:);
end

% compute the delta coefficients
if 1
    delta_coef = 0;
    i = DELTAWINDOW+1;
    denom = sum((1:DELTAWINDOW).^2) * 2;
    for j = 1:DELTAWINDOW
        delta_coef = delta_coef + j/denom*(static_coef(i+j:i+j+N_vec-1,:) - static_coef(i-j:i-j+N_vec-1,:));
    end
else    % implementation 2, use filter function
    weight = [2 1 0 -1 -2]/denom;
    delta_coef = filter(weight, 1, static_coef);
    delta_coef = delta_coef(DELTAWINDOW*2+1:end,:);
end
end