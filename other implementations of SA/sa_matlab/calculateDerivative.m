function deriv = calculateDerivative(evaluateFunction,A,b,bi)

%This function returns the derivative of
%the function evaluated by evaluateFunction(A,b) ,with respect to bi, calculated 
%to a specified point b.
%inputs:
%evaluateFunction :A function handle. Always call like this: calculateDerivative(@evaluateFunction,A,b,bi)
%A : 1x6 cell array containing the matrices A0 .. A5
%b : 1x5 array containing the input of the function evaluated by @evaluateFunction
%bi : An integer between 1..5 indicating the bi with respect to which we we differentiate
%output:
%deriv : the value of the derivative with 3 decimal digits accuracy.

d=0.0000001;
DeltafForward=(evaluateFunction(A,[b(1:bi-1),b(bi)+d,b(bi+1:5)])-evaluateFunction(A,b))/d;
DeltafBackward=(evaluateFunction(A,b)-evaluateFunction(A,[b(1:bi-1),b(bi)-d,b(bi+1:5)]))/d;
    while abs(DeltafForward - DeltafBackward)> 1E-8
        d=d/2;
        DeltafForward=(evaluateFunction(A,[b(1:bi-1),b(bi)+d,b(bi+1:5)])-evaluateFunction(A,b))/d;
        DeltafBackward=(evaluateFunction(A,b)-evaluateFunction(A,[b(1:bi-1),b(bi)-d,b(bi+1:5)]))/d;
    end
deriv=(DeltafForward+DeltafBackward)/2;
end

