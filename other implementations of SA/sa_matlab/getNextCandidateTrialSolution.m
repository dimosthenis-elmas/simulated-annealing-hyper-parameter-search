function [ next ] = getNextCandidateTrialSolution(init_c,initT,currT,b)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


c=init_c*((currT)/initT);

next(1)=b(1)+(c*(-1 + (1+1)*rand(1)));

next(2)=b(2)+(c*(-1 + (1+1)*rand(1)));
    
next(3)=b(3)+(c*(-1 + (1+1)*rand(1)));

next(4)=b(4)+(c*(-1 + (1+1)*rand(1)));

next(5)=b(5)+(c*(-1 + (1+1)*rand(1)));

end

