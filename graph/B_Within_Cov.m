function grad = B_Within_Cov(input_layers, curr_layer, futureLayers)

data = input_layers{1}.a;
label = input_layers{2}.a;

future_grad = GetFutureGrad(futureLayers, curr_layer);


for c = 1:C
    data_c = 1; % get data for class c
    
    B_cov();
    
    B_CMN()
    
end


end


