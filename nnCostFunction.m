function [J grad] = nnCostFunction(nn_params, ...
                                   inputLayerSize, ...
                                   firstHiddenLayerSize, ...
                                   secondHiddenLayerSize, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:firstHiddenLayerSize * (inputLayerSize + 1)), ...
                 firstHiddenLayerSize, (inputLayerSize + 1));

Theta2 = reshape(nn_params((1 + firstHiddenLayerSize * (inputLayerSize + 1)):firstHiddenLayerSize * (inputLayerSize + 1) + secondHiddenLayerSize * (firstHiddenLayerSize + 1)), ...
                 secondHiddenLayerSize, (firstHiddenLayerSize + 1));

Theta3 = reshape(nn_params((1 + firstHiddenLayerSize * (inputLayerSize + 1) + secondHiddenLayerSize * (firstHiddenLayerSize + 1)):end), ...
                 num_labels, (secondHiddenLayerSize + 1));
 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


%
m = size(X, 1);
resy=zeros(m, num_labels);
for i=1:m
	resy(i, y(i))=1;
end
X=[ones(m,1), X];
a1=X;
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1), a2];
z3=a2*Theta2';
a3=sigmoid(z3);
a3=[ones(m,1), a3];
z4=a3*Theta3';
h=sigmoid(z4);
delta4=h-resy;
delta3=(delta4*Theta3);
delta3=delta3(:,2:end).*sigmoidGradient(z3);
delta2=(delta3*Theta2);
delta2=delta2(:,2:end).*sigmoidGradient(z2);
D2=zeros(size(delta4,2),size(a3,2));
D1=zeros(size(delta3,2),size(a2,2));
D0=zeros(size(delta2,2),size(a1,2));
for i=1:m
    D2=D2+(delta4(i,:))'*a3(i,:);
	D1=D1+(delta3(i,:))'*a2(i,:);
	D0=D0+(delta2(i,:))'*a1(i,:);
end
D2=D2/m;
D1=D1/m;
D0=D0/m;
D2(:,2:end)=D2(:,2:end)+(lambda/m)*Theta3(:,2:end);
D1(:,2:end)=D1(:,2:end)+(lambda/m)*Theta2(:,2:end);
D0(:,2:end)=D0(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta1_grad=D0;
Theta2_grad=D1;
Theta3_grad=D2;
j1=sum((resy.*log(h)),'all');
j2=sum(((ones(size(resy))-resy).*log(ones(size(h))-h)),'all');
J=(-1/m)*(j1+j2);
j1=sum((Theta1(:,2:(inputLayerSize+1)).^2),'all');
j2=sum((Theta2(:,2:(firstHiddenLayerSize+1)).^2),'all');
j3=sum((Theta3(:,2:(secondHiddenLayerSize+1)).^2),'all');
J=J+(lambda/(2*m))*(j1+j2+j3);


grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end