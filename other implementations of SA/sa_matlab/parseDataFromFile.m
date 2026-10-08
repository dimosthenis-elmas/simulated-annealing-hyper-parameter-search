function A = parseDataFromFile

%This function parses matrices A0...A5 from the text file matrixdata.txt
%Output:
%A 1x6 cell array containing the matrices A0 to A5

A=cell(1,6);

A0 = dlmread('matrixdata.txt','',[0 0 4 4]);
A{1,1}=A0;

A1 = dlmread('matrixdata.txt','',[5 0 9 4]);
A{1,2}=A1;

A2 = dlmread('matrixdata.txt','',[11 0 15 4]);
A{1,3}=A2;

A3 = dlmread('matrixdata.txt','',[17 0 21 4]);
A{1,4}=A3;

A4 = dlmread('matrixdata.txt','',[23 0 27 4]);
A{1,5}=A4;

A5 = dlmread('matrixdata.txt','',[29 0 33 4]);
A{1,6}=A5;

end

