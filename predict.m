function p = predict(nn_params,secondHiddenLayerSize,firstHiddenLayerSize,inputLayerSize,num_labels, X,y)

Theta1 = reshape(nn_params(1:firstHiddenLayerSize * (inputLayerSize + 1)), ...
                 firstHiddenLayerSize, (inputLayerSize + 1));

Theta2 = reshape(nn_params((1 + firstHiddenLayerSize * (inputLayerSize + 1)):firstHiddenLayerSize * (inputLayerSize + 1) + secondHiddenLayerSize * (firstHiddenLayerSize + 1)), ...
                 secondHiddenLayerSize, (firstHiddenLayerSize + 1));

Theta3 = reshape(nn_params((1 + firstHiddenLayerSize * (inputLayerSize + 1) + secondHiddenLayerSize * (firstHiddenLayerSize + 1)):end), ...
                 num_labels, (secondHiddenLayerSize + 1));

m = size(X, 1);
p = zeros(size(X, 1), 1);
X=[ones(m,1), X];
a2=sigmoid(X*(Theta1)');
a2=[ones(m,1) a2];
a3=sigmoid(a2*(Theta2)');
a3=[ones(m,1) a3];
a4=sigmoid(a3*(Theta3)');
for i=1:m
  [~,index]=max(a4(i,:));
  p(i)=index;
end


end
