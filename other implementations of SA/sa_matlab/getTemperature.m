function [ T ] = getTemperature(initTemp,currentIteration,a)

%This function reduces the temperature of the perevious
%iteration by a factor a

T=initTemp*(a^(currentIteration+1));

end

