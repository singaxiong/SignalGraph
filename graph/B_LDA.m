function grad = B_LDA(input_layers)

Sw = input_layers{1}.a;
St = input_layers{2}.a;




grad{1} = 1; % gradient w.r.t. Sw
grad{2} = 1; % gradient w.r.t. St

end