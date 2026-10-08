function [C] = descend(evaluateFunction,calculateDerivative,A,b,e)

%This function follows the direction of the derivative calculated at b
%to obtain a slightly smaller value than evaluateFunction(A,b) if possible.
%Thus the function 'descends' slightly from the point evaluateFunction(A,b)
%to the point evaluateFunction(A,C{1,1}).
%Inputs:
%evaluateFunction :A function handle. Always call like this: descend(@evaluateFunction,@calculateDerivative,A,b,e)
%calculateDerivative:A function handle.Always call like this: descend(@evaluateFunction,@calculateDerivative,A,b,e)
%A : 1x6 cell array containing the matrices A0 .. A5
%b : 1x5 array containing the input of the function evaluated by evaluateFunction
%e : the factor used to control the descent (b - e*d).See below.A small number.
%Outputs:
%A cell array containing the new point c (1x5 array), and the evaluation of the function at this point C{1,2} 
C=cell(1,2);
for i = 1:5
      d(i)=calculateDerivative(evaluateFunction,A,b,i);
end



C{1,1} = b - e*d;
C{1,2} = evaluateFunction(A,C{1,1});

end

