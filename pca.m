function [avg,U,S] = pca(X)
  X=X/255; 
X=X';
[n m]=size(X);
avg=sum(X,2)/m;
X=X-(sum(X,2))/m;
sigma=X*X';
[u s v]=svd(sigma);
U=u(1:end,1:300); 
S=s;

  