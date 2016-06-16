 function [OutStruct] = imsetup(t60,room,distance,angle)

% PURPOSE       : Set environmental parameters for imaging RIR simulation
% Edit the parameters to generate your own imaging RIR. The function
% returns the structure 'outStruct' with all simulatin parameters.

% AUTHOR        : Xionghu Zhong
% DATE          : Oct. 4, 2006  Modified on sep. 2009
% EMAIL         : x.zhong@ed.ac.uk, zxh@ieee.org

OutStruct.soundVelocity = 343;                        % propagation speed of acoustic waves in m/s
OutStruct.samplingFreq = 16000;                        % reset the sampling frequency
OutStruct.micNumber = 8;      
OutStruct.t60 = t60;
OutStruct.rirLength = t60*OutStruct.samplingFreq;

switch upper(room)
    case 'SMALL'
        OutStruct.roomDimension = [7.00 5.00 3.00];           % room dimensions in m
    case'MEDIUM'
        OutStruct.roomDimension = [12.00 10.00 3.00];           % room dimensions in m    
    case 'LARGE'
        OutStruct.roomDimension = [17.00 15.00 3.00];       % original [15,10,3]   % room dimensions in m    
    otherwise error('Room size is not defined...');
end;
LL = OutStruct.roomDimension(1,1);
WW = OutStruct.roomDimension(1,2);
HH = OutStruct.roomDimension(1,3);

if isequal(t60,0)
    beta = 0;
else
%     alpha = 0.161*prod(OutStruct.roomDimension)/(2*t60*(LL*WW+LL*HH+WW*HH));
%     beta = sqrt(1-alpha);
    beta = exp(-13.82/((1/LL+1/WW+1/HH)*OutStruct.soundVelocity*t60));%calculate the reflection coefficient
end;

OutStruct.reflectCoeffs = [ones(1,6)*beta];           % reflection coefficients in range [0 ... 1]

OutStruct.arrayDiameter = 0.2;
r = OutStruct.arrayDiameter/2; % radius of the circular array
deltaAngle = 2*pi/OutStruct.micNumber;
micAngle = 0:deltaAngle:deltaAngle*(OutStruct.micNumber-1);%2*pi*rand(OutStruct.microNumber/2,1);
deltaX = r.*sin(micAngle)';
deltaY = r.*cos(micAngle)';
deltaZ = -0.5.*ones(OutStruct.micNumber,1);% array height 0.5m below the center point
micCenter = OutStruct.roomDimension./2;
OutStruct.micPosition = repmat(micCenter,OutStruct.micNumber,1) + [deltaX deltaY deltaZ]; % This changes the area where the nodes are scattered.


switch upper(distance)
    case 'FAR'
        if isequal(room,'small'), 
            R=2;  % distance between the source and the microphone array
        elseif isequal(room, 'medium')
            R=4;
        else
            R=6.5;
        end;      
    case'NEAR'
        R= 1;
    otherwise error('Source distance is not defined...');
end;

% I = find(R>min(OutStruct.roomDimension(1:2)./2)-0.1||R<0.5);
% while ~isempty(I)
%     R(I) = R1+R0.*randn(length(I),1);
%     I = find(R>min(OutStruct.roomDimension(1:2)./2)-0.1||R<0.5);  % the distance to the walls should be larger than 10cm
% end;
OutStruct.dist = R;
%OutStruct.distRandn = R;
%OutStruct.Angle = Angle;
%OutStruct.sourceAngleRandn = OutStruct.sourceAngle;% + 0.5.*randn(1,360);
srcAngle = angle*pi/180;
OutStruct.srcPosition = micCenter+[R*sin(srcAngle) R*cos(srcAngle) 0];

% 
% figure(100)
% h0=plot(micCenter(1,1), micCenter(1,2),'ko');hold on;
% h1=plot(OutStruct.micPosition(:,1),OutStruct.micPosition(:,2),'bo');hold on;
% h2=plot(OutStruct.srcPosition(:,1),OutStruct.srcPosition(:,2),'r--');hold on;
% set(h2,'lineWidth',2);
% % h3=plot3(outStruct.x2,outStruct.y2,outStruct.z2,'r-');hold on;
% % set(h3,'lineWidth',2)
% xlabel('x/m')
% ylabel('y/m')
% %zlabel('z')
% grid on
%axis([0 OutStruct.roomDimension(1,1) 0 OutStruct.roomDimension(1,2)]);
%view(-50,10)
%legend([],'microphone','source position');

