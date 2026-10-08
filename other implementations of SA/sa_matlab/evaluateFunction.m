function [ Z ] = evaluateFunction(A,b)

%This function evaluates f(b)=ëmax(A0+Sum(biAi)) for a given b

%Inputs:
%A: 1x6 cell array containing matrices A0 to A5
%b: 1x5 matrix containing the bi values
%Output :
%The maximum eigenvalue measure of the matrix A0+Sum(biAi)

Z=A{1,1};  
  for i = 2:6
      Z = Z+b(i-1)*A{1,i};
  end
  Z = max(abs(eig(Z)));
end

