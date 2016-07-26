function [model,y] = my_kmeans(X,num_centers,display, init_center)
% KMEANS K-means clustering algorithm.
% 
% Synopsis:
%  [model,y] = kmeans(X,num_centers)
%  [model,y] = kmeans(X,num_centers,Init_centers)
%
% Description:
%  [model,y] = kmeans(X,num_centers) runs K-means clustering 
%   where inital centers are randomly selected from the 
%   input vectors X. The output are found centers stored in 
%   structure model.
%   
%  [model,y] = kmeans(X,num_centers,Init_centers) uses
%   init_centers as the starting point.
%
% Input:
%  X [dim x num_data] Input vectors.
%  num_centers [1x1] Number of centers.
%  Init_centers [1x1] Starting point of the algorithm.
%    
% Output:
%  model [struct] Found clustering:
%   .X [dim x num_centers] Found centers.
%
%   .y [1 x num_centers] Implicitly added labels 1..num_centers.
%   .t [1x1] Number of iterations.
%   .MsErr [1xt] Mean-Square error at each iteration.
%
%  y [1 x num_data] Labels assigned to data according to 
%   the nearest center.
%
% Example:
%  data = load('riply_trn');
%  [model,data.y] = kmeans( data.X, 4 );
%  figure; ppatterns(data); 
%  ppatterns(model,12); pboundary( model );
%
% See also 
%  EMGMM, KNNCLASS.
%

% (c) Statistical Pattern Recognition Toolbox, (C) 1999-2003,
% Written by Vojtech Franc and Vaclav Hlavac,
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>,
% <a href="http://www.feld.cvut.cz">Faculty of Electrical engineering</a>,
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

% Modifications:
% 12-may-2004, VF

[dim,num_data] = size(X);

if nargin<4
    % random inicialization of class centers
    inx=randperm(num_data);
    model.X = X(:,inx(1:num_centers));  
else
    model.X = init_center;
end
 model.y = 1:num_centers;
 model.K = 1;
    
if nargin < 3,
    display = 1;
end

model.fun = 'knnclass';

old_y = zeros(1,num_data);
t = 0;

%xiao 
iteration = 0;

% main loop
%-------------------------
while 1,
  
  t = t+1;
 
  % classificitation
    %   y = knnclass( X, model );
  if 0
      for i=1:num_centers
          tmp = X - repmat(model.X(:,i), 1, num_data);
          if size(tmp,1)==1
              distance(i,:) = tmp.*tmp;
          else
              distance(i,:) = sum(tmp.*tmp);
          end
      end
      [tmp, y] = min(distance);

  else  % if we have big memory, we can do the distance computation in one short
      BlockSize = 1e5;
      y = zeros(1,size(X,2));
      for i=1:BlockSize:size(X,2)
          idx = min(size(X,2),i+BlockSize-1);
          distance = bsxfun(@plus, sum(model.X.^2,1)', sum(X(:,i:idx).^2,1)) - 2*model.X'*X(:,i:idx);
          [tmp, y(i:idx)] = min(distance);
      end
  end
 

  % computation of class centers
  err = 0;
  for i=1:num_centers,
    inx = find(y == i);

    if ~isempty(inx),
      
      % compute approximation error
      err = err + sum(sum((X(:,inx) - model.X(:,i)*ones(1,length(inx)) ).^2));
      
      % compute new centers
      model.X(:,i) = sum(X(:,inx),2)/length(inx);
    end
  end

  % Number of iterations and Mean-Square Error 
  model.t = t;
  model.MsErr(t) = err/num_data;
  
  % xiao 
  if display
      text = sprintf('Iteration %d, MSE=%f, %s',t,model.MsErr(t), datestr(now));
      disp(text);
  end
  if sum( abs(y - old_y) ) == 0,
    return;
  end
  if t>1
      if abs(model.MsErr(t)-model.MsErr(t-1))/model.MsErr(t-1) <0.002
          return;
      end
  end

  old_y = y;
end

return;
% EOF
