function [X_Init]=ProjectData(X,principal)
  X=X/255;  %changed now
  X=X';
  [~ , m] =size(X);
  X=X-(sum(X,2))/m;
  X_Init=(principal')*(X);
  X_Init=X_Init';         