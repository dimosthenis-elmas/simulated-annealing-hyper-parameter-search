

clear %clear entire workspace to avoid an OUT OF MEMORY warning
A=parseDataFromFile;  %parse Ai matrices from file

initTemp=900; %initial temperature
currentTemp = initTemp;
max_iterations=5000; %total  number of iterations


history=zeros(max_iterations); %To keep track of the cost history
tempHistory=zeros(max_iterations); %To keep track of the temperature

init_c = 0.05;   %initial value for the disturbance factor c
a=0.997; %temperature reduction factor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin simulated annealing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


currentTrialSolution = [-0.8302 -0.5785 0.0126 0.0144 -0.0517]; %the initial trial solution to begin the simulation from
fprintf('The initial trial solution is:');
fprintf('\t\t%.4f',currentTrialSolution);
fprintf('\n');
fprintf('The cost of the initial trial solution is:');
fprintf('\t\t%.4f',evaluateFunction(A,currentTrialSolution));
fprintf('\n');


for i=1:max_iterations
  d=0.005;
  %Descend from current solution a bit for 5 times. Take care of the step d so that you
  %do not get a higher cost.If so reduce the step d and try again.
  for j=1:5
    C = descend(@evaluateFunction,@calculateDerivative,A,currentTrialSolution,d);
    if(C{1,2}<evaluateFunction(A,currentTrialSolution))
        currentTrialSolution = C{1,1};
    else
        d=d/2;
        C = descend(@evaluateFunction,@calculateDerivative,A,currentTrialSolution,d);
        while C{1,2} > evaluateFunction(A,currentTrialSolution)
            d=d/2;
            C = descend(@evaluateFunction,@calculateDerivative,A,currentTrialSolution,d);
        end
        currentTrialSolution = C{1,1};
    end
  end

  nextTrialSolution = getNextCandidateTrialSolution(init_c,initTemp,currentTemp,currentTrialSolution);


  currentCost = evaluateFunction(A,currentTrialSolution);
  nextCost = evaluateFunction(A,nextTrialSolution);
  diff = abs(currentCost-nextCost);
  if nextCost<currentCost
      currentTrialSolution = nextTrialSolution;
      currentCost = nextCost;
  else
      if rand(1)<exp(-diff/(currentTemp))
         currentTrialSolution = nextTrialSolution;
         currentCost = nextCost;
      end
  end
  %Reduce the temperature
  currentTemp = getTemperature(initTemp,i,a);
  %Store the history
  tempHistory(i)=currentTemp;
  history(i)=currentCost;

 %print sub-results for every 100 iterations
 if mod(i,100)== 0
    i
    currentTrialSolution
    fprintf('Cost: ');
    fprintf('\t\t%.4f',history(i));
    fprintf('\n');
 end
end

%Plot the costs and the temperature
figure
subplot(2,1,1)
plot(history);
subplot(2,1,2)
plot(tempHistory);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End simulated annealing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Print results
fprintf('The final solution is:');
fprintf('\t\t%.4f',currentTrialSolution);
fprintf('\n');
fprintf('The cost of the final solution is:');
fprintf('\t\t%.4f',evaluateFunction(A,currentTrialSolution));
fprintf('\n');
