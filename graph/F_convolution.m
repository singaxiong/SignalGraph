function output = F_convolution(input, weights, b)
[D1, D2, nChannel, nImage] = size(input);
[F1, F2, nChannel2, nFeature] = size(weights);

if nChannel ~= nChannel2
    fprintf('Error: number of input feature maps is not equal to the number of feature maps from previous layer!\n');
end

output = zeros(D1-F1+1, D2-F2+1, nFeature, nImage);

for imageNum = 1:nImage
    
    for featureNum = 1:nFeature
        
        % convolution of image with feature matrix for each channel
        convolvedImage = zeros(D1-F1+1, D2-F2+1);
        
        for channel = 1:nChannel
            
            feature = weights(:,:,channel, featureNum);
            % Flip the feature matrix because of the definition of convolution, as explained later
            feature = flipud(fliplr(squeeze(feature)));
            
            % Obtain the image
            im = squeeze(input(:, :, channel, imageNum));
            convolvedImage = convolvedImage + conv2(im, feature, 'valid');
        end
        
        output(:,:,featureNum, imageNum) = convolvedImage + b(featureNum);
    end
end

end

