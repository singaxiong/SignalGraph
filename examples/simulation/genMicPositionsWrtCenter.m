function [mic] = genMicPositionsWrtCenter(rotate)
DEBUG = 0;

r = 10; % radius of the circular array in cm
% use the array center as the origin of the coordinate system
%
%             1
%       8           2
%
%     7       o       3  -----> x axis
%
%       6           4
%             5

% define the angle between mics and the x axis
mic_angle = [90:-45:-225] + rotate;
mic_angle = mic_angle/180*pi;
xmic = [cos(mic_angle)];
ymic = [sin(mic_angle)];
zmic = zeros(size(xmic));
mic = [xmic; ymic; zmic] * r / 100;

if DEBUG
    plot(mic(1,:), mic(2,:), '*'); 
    for i=1:size(mic,2)
        text(mic(1,i)+0.01, mic(2,i), num2str(i));
    end
end

end