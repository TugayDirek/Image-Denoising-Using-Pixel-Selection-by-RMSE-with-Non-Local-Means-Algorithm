clear all; close all; clc;


img = imread("a2.png");

%convert RGB image to grey image
[rows_im, columns_im, numberOfColorChannels_im] = size(img);
if numberOfColorChannels_im>1
    img = rgb2gray(img);
end
img = double(img);

%specify search size and kernel size, pad rows and cols to image
targetSize = 9;
searchSize = 15;
search_pad_size = (searchSize-1)/2;
target_pad_size = (targetSize-1)/2;
padSize = searchSize;
img2 = padarray(img,[padSize padSize],"symmetric");
[rows, cols] = size(img2);

output = zeros(size(img));%for nl-means with pixel selection
output2 = zeros(size(img));%for original nl-means 
counter = 0;

h = 55; %you can play with this parameter for different results
lambda = 0.95;
% Traverse all pixels of the image
for i = 1+padSize:rows-padSize
    for j = 1+padSize:cols-padSize
        
        counter = counter +1;

        % current kernel to be processed
        current = img2(i-target_pad_size:i+target_pad_size, j-target_pad_size:j+target_pad_size);
        
        
        weights = zeros([searchSize searchSize]);
        weights2 = zeros([searchSize searchSize]);
        sumWeights = 0;
        sumWeights2 = 0;

        rmse_sum =0;
        rmse_counter = 0;

        % define search area and pad
        search_area = img2(i-search_pad_size:i+search_pad_size,j-search_pad_size:j+search_pad_size);
        padded_search_area = padarray(search_area,[target_pad_size target_pad_size],"symmetric");
        [rows2, cols2] = size(padded_search_area);


        tic;
        % Loop over the search window
         
        for k = 1+target_pad_size:rows2-target_pad_size
            for m = 1+target_pad_size:cols2-target_pad_size

                % Compute the squared distance between the current patch and the patch at (x+dx, y+dy)
                target = padded_search_area(k-target_pad_size:k+target_pad_size, m-target_pad_size:m+target_pad_size);
                OneDTarget = reshape(target',[1 size(target,1)*size(target,2)]);

                X = ones(size(OneDTarget));
                [rows_target,cols_target] = size(OneDTarget);
                for l=1:cols_target
                    X(l) = l;
                end

                mdl = fitlm(X,OneDTarget);
                mse = mdl.MSE;
                rmse = mdl.RMSE;
                rmse_sum = rmse_sum + rmse;
                rmse_counter = rmse_counter + 1;
                %disp("---");
                %disp(mse);
                %disp(rmse);
                %disp("///");              
            end
        end
        
        rmse_avg = rmse_sum/rmse_counter;
        disp(counter);
        toc;
        tic;

        % traverse through all pixels in search area
        for k = 1+target_pad_size:rows2-target_pad_size
            for m = 1+target_pad_size:cols2-target_pad_size
                
                % Compute the squared distance between the current patch and the patch at (x+dx, y+dy)
                target = padded_search_area(k-target_pad_size:k+target_pad_size, m-target_pad_size:m+target_pad_size);
                OneDTarget = reshape(target',[1 size(target,1)*size(target,2)]);

                X = ones(size(OneDTarget));
                [rows_target,cols_target] = size(OneDTarget);
                for l=1:cols_target
                    X(l) = l;
                end

                mdl = fitlm(X,OneDTarget);
                mse = mdl.MSE;
                rmse = mdl.RMSE;
               
                weight=0;
                if(rmse<rmse_avg)
                    dist = sum((current(:) - target(:)).^2);
                    % Compute the weight for this patch using the non-local means formula
                    weight = exp(-dist / (h^2));
                    %disp(k);disp(m);
                    % Update the weights and sum of weights
                end

                

                % for original nl-means
                % calculate the distance between two kernels
                dist2 = sum((current(:) - target(:)).^2);
                % implement NL-means formula
                weight2 = exp(-dist2 / (h^2));

                % save the weights for each pixel and sum all of them
                weights2(k-target_pad_size, m-target_pad_size) = weight2;
                sumWeights2 = sumWeights2 + weight2;

                weights(k-target_pad_size, m-target_pad_size) = weight;
                sumWeights = sumWeights + weight;
            end
        end
        disp(counter);
        toc;
        weights = weights / sumWeights;
        weights2 = weights2 / sumWeights2;

        
        % calculate the weighted average of all pixels in search area
        output(i,j) = sum(weights(:) .* search_area(:));
        output2(i,j) = sum(weights2(:) .* search_area(:));

    end
end

output = output(padSize+1:rows-padSize,padSize+1:cols-padSize);
output2 = output2(padSize+1:rows-padSize,padSize+1:cols-padSize);

%noisy image, nl-means with pixel selection and original nl-means
figure;imshow([uint8(img) uint8(output) uint8(output2)]);

%calculate psnr values
peaksnr_nl_means = psnr(uint8(img),uint8(output));
peaksnr_pixel_selection = psnr(uint8(img),uint8(output2));






