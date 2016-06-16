function output = F_cosine(input_layers)

input1 = input_layers{1}.a;
input2 = input_layers{2}.a;


output = sum(input1 .* input2);

norm1 = sqrt(diag(input1'*input1))';
norm2 = sqrt(diag(input2'*input2))';
% for i=1:size(input1,2)
%     norm1a(i) = norm(input1(:,i));
% end
output = output ./ (norm1 .* norm2);


end


% function output = F_cosine(input_layers)
% 
% input1 = input_layers{1}.a;
% input2 = input_layers{2}.a;
% 
% 
% output = sum(input1 .* input2);
% 
% norm1 = sqrt(diag(input1'*input1))';
% norm2 = sqrt(diag(input2'*input2))';
% % for i=1:size(input1,2)
% %     norm1a(i) = norm(input1(:,i));
% % end
% output = output ./ (norm1 .* norm2);
% 
% 
% end

% % Old and slow implementation
% function output = F_cosine(input_layers)
% 
% input1 = input_layers{1}.a;
% input2 = input_layers{2}.a;
% 
% 
% output = sum(input1 .* input2);
% 
% % norm1 = sqrt(diag(input1'*input1))';
% % norm2 = sqrt(diag(input2'*input2))';
% 
% norm1 = zeros(1,size(input1,2));
% for i=1:size(input1,2)
%     norm1(i) = norm(input1(:,i));
% end
% 
% norm2 = zeros(1,size(input1,2));
% for i=1:size(input1,2)
%     norm2(i) = norm(input2(:,i));
% end
% 
% output = output ./ (norm1 .* norm2);
% 
% 
% end
