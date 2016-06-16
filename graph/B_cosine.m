function [grad] = B_cosine(input_layers, future_layers)

future_grad = 0;
for i=1:length(future_layers)
    future_grad = future_grad + future_layers{i}.grad;
end

input1 = input_layers{1}.a;
input2 = input_layers{2}.a;
% dim = size(input1,1);
norm1 = sqrt(diag(input1'*input1))';
norm2 = sqrt(diag(input2'*input2))';

norm_inverse = 1./(norm1.*norm2);
tmp_grad = bsxfun(@times, input2, norm_inverse);
atb = diag(input1'*input2)';
tmp_grad = tmp_grad - bsxfun(@times, input1, atb./((norm1.^3).*norm2));

tmp_grad2 = bsxfun(@times, input1, norm_inverse);
atb2 = diag(input2'*input1)';
tmp_grad2 = tmp_grad2 - bsxfun(@times, input2, atb2./((norm2.^3).*norm1));

% n_input1 = norm(input1);
% n_input2 = norm(input2);

% new_input1 = ((eye(dim)/n_input1)-((input1*(input1'))/(n_input1).^3))*(input2/n_input2);
% new_input2 = ((eye(dim)/n_input2)-((input2*(input2'))/(n_input2).^3))*(input1/n_input1);

% new_input1 = (input2/(n_input1*n_input2))-(input1*(input1'*input2)/(n_input2*(n_input1).^3));
% new_input2 = (input1/(n_input1*n_input2))-(input2*(input2'*input1)/(n_input1*(n_input2).^3));

grad{1} = bsxfun(@times, tmp_grad, future_grad);
grad{2} = bsxfun(@times, tmp_grad2, future_grad);

end

% function [grad, grad2] = B_cosine(input_layers, future_layers)
% 
% future_grad = 0;
% for i=1:length(future_layers)
%     future_grad = future_grad + future_layers{i}.grad;
% end
% 
% input1 = input_layers{1}.a;
% input2 = input_layers{2}.a;
% % dim = size(input1,1);
% norm1 = sqrt(diag(input1'*input1))';
% norm2 = sqrt(diag(input2'*input2))';
% 
% norm_inverse = 1./(norm1.*norm2);
% tmp_grad = bsxfun(@times, input2, norm_inverse);
% atb = diag(input1'*input2)';
% tmp_grad = tmp_grad - bsxfun(@times, input1, atb./((norm1.^3).*norm2));
% 
% tmp_grad2 = bsxfun(@times, input1, norm_inverse);
% atb2 = diag(input2'*input1)';
% tmp_grad2 = tmp_grad2 - bsxfun(@times, input2, atb2./((norm2.^3).*norm1));
% 
% % n_input1 = norm(input1);
% % n_input2 = norm(input2);
% 
% % new_input1 = ((eye(dim)/n_input1)-((input1*(input1'))/(n_input1).^3))*(input2/n_input2);
% % new_input2 = ((eye(dim)/n_input2)-((input2*(input2'))/(n_input2).^3))*(input1/n_input1);
% 
% % new_input1 = (input2/(n_input1*n_input2))-(input1*(input1'*input2)/(n_input2*(n_input1).^3));
% % new_input2 = (input1/(n_input1*n_input2))-(input2*(input2'*input1)/(n_input1*(n_input2).^3));
% 
% grad = bsxfun(@times, tmp_grad, future_grad);
% grad2 = bsxfun(@times, tmp_grad2, future_grad);
% 
% end

% % Old and slow implementation
% future_grad = 0;
% for i=1:length(future_layers)
%     future_grad = future_grad + future_layers{i}.grad;
% end
% 
% input1 = input_layers{1}.a;
% input2 = input_layers{2}.a;
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
% tmp_grad = zeros(size(input1,1),size(input1,2));
% tmp_grad2 = zeros(size(input1,1),size(input1,2));
% for i=1:size(input1,2),
%     tmp_grad(:,i) = (input2(:,i)/(norm1(i)*norm2(i)))-input1(:,i)*((input1(:,i)'*input2(:,i))/((norm1(i).^3)*norm2(i)));
%     tmp_grad2(:,i) = (input1(:,i)/(norm1(i)*norm2(i)))-input2(:,i)*((input2(:,i)'*input1(:,i))/((norm2(i).^3)*norm1(i)));
% end






