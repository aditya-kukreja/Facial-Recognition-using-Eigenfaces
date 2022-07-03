

clear ; close all; clc
load YaleB_32x32.mat
[~,principal,~]=pca(fea(1:2:2414,:));
X_Train=ProjectData(fea(1:2:2414,:),principal);
X_Test=ProjectData(fea(2:2:2414,:),principal);
y=gnd;
y_Train=y(1:2:2414);
y_Test=y(2:2:2414);


inputLayerSize  = 300; 
firstHiddenLayerSize = 100;
secondHiddenLayerSize=50;
num_labels = 38;          % 38 labels 
                         
m = size(X_Train, 1);

Theta1_Initialized = randInitializeWeights(inputLayerSize, firstHiddenLayerSize);
Theta2_Initialized = randInitializeWeights(firstHiddenLayerSize,secondHiddenLayerSize);
Theta3_Initialized = randInitializeWeights(secondHiddenLayerSize, num_labels);

initial_nn_params = [Theta1_Initialized(:) ; Theta2_Initialized(:) ; Theta3_Initialized(:);];

options = optimset('MaxIter', 1000);


lambda = 1.5;


costFunction = @(p) nnCostFunction(p, ...
                                   inputLayerSize, ...
                                   firstHiddenLayerSize, ...
                                   secondHiddenLayerSize, ...
                                   num_labels, X_Train, y_Train, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


Theta1 = reshape(nn_params(1:firstHiddenLayerSize * (inputLayerSize + 1)), ...
                 firstHiddenLayerSize, (inputLayerSize + 1));

Theta2 = reshape(nn_params((1 + firstHiddenLayerSize * (inputLayerSize + 1)):firstHiddenLayerSize * (inputLayerSize + 1) + secondHiddenLayerSize * (firstHiddenLayerSize + 1)), ...
                 secondHiddenLayerSize, (firstHiddenLayerSize + 1));

Theta3 = reshape(nn_params((1 + firstHiddenLayerSize * (inputLayerSize + 1) + secondHiddenLayerSize * (firstHiddenLayerSize + 1)):end), ...
                 num_labels, (secondHiddenLayerSize + 1));

save trainedParams.mat nn_params;
pred = predict(nn_params,secondHiddenLayerSize,firstHiddenLayerSize,inputLayerSize,num_labels,X_Test,y_Test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_Test)) * 100);


