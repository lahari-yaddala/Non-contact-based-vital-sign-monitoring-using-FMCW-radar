MATLAB CODES: 
1)CODE FOR BIN TO MAT FILE CONVERSION: 
% --- Load Recording Parameters --- load('iqData_RecordingParameters.mat'); 
% --- Important Variables --- numADCSamples = RecordingParameters.SamplesPerChirp; numRX = RecordingParameters.NumReceivers; numChirps = RecordingParameters.NumChirps; 
% --- Read the BIN file --- fid = fopen('slow.bin', 'r'); adcData = fread(fid, 'int16'); fclose(fid); 
% --- Form complex data (I + jQ) --- adcDataComplex = adcData(1:2:end) + 1i * adcData(2:2:end); 
% --- Calculate total chirps --- totalChirps = floor(length(adcDataComplex) / (numADCSamples * numRX)); 
% --- Truncate the extra data if needed --- adcDataComplex = adcDataComplex(1 : totalChirps * numADCSamples * numRX); 
% --- Reshape into [numSamples, numRX, numChirps] --- adcDataComplex = reshape(adcDataComplex, [numADCSamples, numRX, totalChirps]); adcDataComplex = permute(adcDataComplex, [3,2,1]);  % [Chirps × RX × Samples] 
% --- Save into a MAT file --- save(['matfiles'], 'adcDataComplex'); 
 
2)MATFILES TO SPECTOGRAMS GENERATION: clc; clear; close all; 
 Folder containing all MAT files dataFolder = "D:\seperated_matfiles\slow"; files = dir(fullfile(dataFolder, "*.mat")); 
% Radar parameters samplesPerChirp = 256; numChirps = 256; fs = 10;   % Frame rate (Hz) – you had fs = 1/0.1, same as 10 Hz for k = 1:length(files)     filename = fullfile(dataFolder, files(k).name);     disp("Processing file: " + files(k).name); 
    % Load IQ data 
    S = load(filename);     raw = S.adcDataComplex;  % [samples × RX × chirps]     % --- Verify data shape ---     totalSamples = numel(raw);     expected = samplesPerChirp * numChirps;     if mod(totalSamples, expected) ~= 0 
        % Crop extra samples that don't fit exactly into full frames         validLength = floor(totalSamples / expected) * expected;         raw = raw(1:validLength);     end 
    % --- Reshape only if needed --- 
    % Sometimes adcDataComplex is already 3D [samples x RX x chirps]     % so we reshape carefully:     raw = reshape(raw, samplesPerChirp, numChirps, []); 
    % ---- RANGE FFT ----     rangeFFT = fft(raw, 1024, 1);     powerRange = mean(abs(rangeFFT), 3);   % average across RX 
    [~, chestBin] = max(mean(abs(powerRange), 2));  % strongest range bin 
    % ---- PHASE EXTRACTION ----     chestSignal = squeeze(rangeFFT(chestBin, :, :));     chestPhase = unwrap(angle(mean(chestSignal, 2))); 
 
    % ---- BANDPASS FILTERS ----     % Breathing: 0.1 - 0.5 Hz     breath = bandpass(chestPhase, [0.1 0.5], fs); 
    % Heart: 0.8 - 2.0 Hz     heart = bandpass(chestPhase, [0.8 2.0], fs); 
    % ---- SPECTROGRAMS ----     % Breathing spectrogram     figure('Visible','off');     spectrogram(breath, 128, 120, 256, fs, 'yaxis');     title("Breathing Spectrogram - " + files(k).name); 
    saveas(gcf, fullfile(dataFolder, files(k).name + "_breath.png")); 
    % Heart spectrogram     figure('Visible','off');     spectrogram(heart, 128, 120, 256, fs, 'yaxis');     title("Heart Spectrogram - " + files(k).name);     saveas(gcf, fullfile(dataFolder, files(k).name + "_heart.png"));     close all; end 
disp(" Processing Complete. Spectrograms saved as PNG."); 
 
3)ML MODEL: 
clc; clear; close all; 
------------------ PATH SETUP ------------------ dataFolder = "D:\Final Dataset"; % Folder containing 6 subfolders newImagePath = "D:\Final Dataset\Heart_Slow\slow_p2.mat_heart.png"; % Path of a new spectrogram to classify 
------------------ LOAD DATASET ------------------ inputSize = [227 227 3]; % CNN input size  Image datastore with auto-resize function imds = imageDatastore(dataFolder, ... 
    'IncludeSubfolders', true, ... 
    'LabelSource', 'foldernames', ... 
    'ReadFcn', @(x)imresize(imread(x), inputSize(1:2)));  Show class counts disp("Class labels and counts:"); countEachLabel(imds) 
------------------ TRAIN / TEST SPLIT ------------------ 
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');  ------------------ DEFINE CNN ARCHITECTURE ------------------ layers = [     imageInputLayer(inputSize, 'Name', 'input')     convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_1')     batchNormalizationLayer('Name', 'bn_1')     reluLayer('Name', 'relu_1')     maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')     convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_2')     batchNormalizationLayer('Name', 'bn_2')     reluLayer('Name', 'relu_2')     maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')     convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_3')     batchNormalizationLayer('Name', 'bn_3')     reluLayer('Name', 'relu_3')     maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')     fullyConnectedLayer(6, 'Name', 'fc') % 6 classes     softmaxLayer('Name', 'softmax')     classificationLayer('Name', 'output') 
]; 
------------------ TRAINING OPTIONS ------------------ options = trainingOptions('adam', ... 
    'InitialLearnRate', 1e-4, ... 
    'MaxEpochs', 20, ... 
    'MiniBatchSize', 32, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', imdsTest, ... 
    'ValidationFrequency', 5, ... 
    'Verbose', false, ... 
    'Plots', 'training-progress'); 
------------------ TRAIN THE CNN ------------------ disp(" Training started..."); 
net = trainNetwork(imdsTrain, layers, options); 
 ------------------ EVALUATE PERFORMANCE ------------------ 
YPred = classify(net, imdsTest); YTest = imdsTest.Labels; accuracy = mean(YPred == YTest); disp(" Test Accuracy: " + accuracy*100 + "%"); figure; confusionchart(YTest, YPred); title('Confusion Matrix - Test Data'); 
 ------------------ PREDICT NEW IMAGE ------------------ if isfile(newImagePath)     img = imread(newImagePath);     img = imresize(img, inputSize(1:2));     predictedLabel = classify(net, img);     figure;      imshow(img);     title("Predicted Class: " + string(predictedLabel));     disp(" Predicted Class: " + string(predictedLabel)); else    warning("New image path not found: " + newImagePath); end disp(" Classification complete!"); 
 
 
