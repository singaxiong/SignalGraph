function output = F_Within_Cov(input_layers)

data = input_layers{1}.a;
label = input_layers{2}.a;

for c = 1:C
    data_c = 1; % get data for class c
    
      
    fake_layer.a = data_c;
    data_c2 = CMN(fake_layer')';
    cov_c(:,:,c) = cov(data_c2');
    
end

output = mean(cov_c,3);

end


