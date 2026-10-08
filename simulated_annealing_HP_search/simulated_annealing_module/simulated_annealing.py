import random
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# The history array holds the scores of previously examined cells from
# the grid of parameters. This is used so that we do not fit the model
# again, for combinations of hyper parameters that have already been exanined
# previously.
@static_vars(h=[])
def history():
    return


def add_to_history(x,y, val):
    history.h[x][y] = val


def search_history(x,y):
    return history.h[x][y]


def init_history(x,y):
    history.h = np.zeros((x,y))

@static_vars(counter=0)
def evals_counter():
    evals_counter.counter += 1

def get_evals_counter():
    return evals_counter.counter


def get_SA_history():
    return history.h


def evaluate_function(map_coords, param_grid,x_train, y_train, cv=5):
    scoring = ''
    # if already evaluated before
    if search_history(map_coords[0], map_coords[1]) > 0:
        return search_history(map_coords[0], map_coords[1])

    p = {'C': [],'gamma': [],'kernel': ['rbf',]}
    p['C'] = [param_grid['C'][map_coords[0]],]
    p['gamma'] = [param_grid['gamma'][map_coords[1]],]

    clf = GridSearchCV(SVC(), p, scoring=['f1_macro'], refit='f1_macro', verbose=0, cv=cv, n_jobs=-1)
    evals_counter()
    clf.fit(x_train, y_train)
    add_to_history(map_coords[0], map_coords[1], clf.best_score_)
    return clf.best_score_

def get_temperature(initTemp, currentIteration, a):
    return initTemp*(a**(currentIteration+1))


def ascend(grid_coords, x_train, y_train, param_grid, cv):
    Center_cell = {'coords': [grid_coords[0], grid_coords[1]], 'score': 0.0}
    E_cell = {'coords': [grid_coords[0]+1, grid_coords[1]], 'score': 0.0}
    NE_cell = {'coords': [grid_coords[0]+1, grid_coords[1]+1], 'score': 0.0}
    SE_cell = {'coords': [grid_coords[0]+1, grid_coords[1]-1], 'score': 0.0}
    N_cell = {'coords': [grid_coords[0], grid_coords[1]+1], 'score': 0.0}




    S_cell = {'coords': [grid_coords[0], grid_coords[1]-1], 'score': 0.0}
    SW_cell = {'coords': [grid_coords[0]-1, grid_coords[1]-1], 'score': 0.0}
    W_cell = {'coords': [grid_coords[0]-1, grid_coords[1]], 'score': 0.0}
    NW_cell = {'coords': [grid_coords[0]-1, grid_coords[1]+1], 'score': 0.0}

    cell_list = [Center_cell, E_cell, NE_cell,SE_cell,N_cell,S_cell,SW_cell,W_cell,NW_cell ]

    # remove cells that got ouf of the grid's bounds
    cell_list = list(filter(lambda e: e['coords'][0] < len(param_grid['C']), cell_list))
    cell_list = list(filter(lambda e: e['coords'][0] > 0, cell_list))
    cell_list = list(filter(lambda e: e['coords'][1] < len(param_grid['gamma']), cell_list))
    cell_list = list(filter(lambda e: e['coords'][1] > 0 , cell_list))

    for e in cell_list:
        e['score'] = evaluate_function(e['coords'], param_grid, x_train, y_train, cv)

    maxScoreCell = max(cell_list, key=lambda c: c['score'])

    return maxScoreCell['coords'], maxScoreCell['score']



def get_next_candidate(init_c, initT, currT, grid_coords, param_grid):
    #print("Temp : %d" % currT)
    c = init_c*(currT/initT)
    new_coords = grid_coords
    disturbance = round(c * (random.uniform(-1, 1)))
    new_C_index = new_coords[0] + disturbance
    while(new_C_index > (len(param_grid['C'])-1)) or (new_C_index < 0):
        disturbance = round(c * (random.uniform(-1, 1)))
        new_C_index = new_coords[0] + disturbance

    disturbance = round(c * (random.uniform(-1, 1)))
    new_gamma_index = new_coords[1] + disturbance
    while (new_gamma_index > (len(param_grid['gamma'])-1)) or (new_gamma_index < 0):
        disturbance = round(c * (random.uniform(-1, 1)))
        new_gamma_index = new_coords[1] + disturbance

    return [new_C_index, new_gamma_index]


def simulated_annealing(starting_point, x_train, y_train, params_grid, max_iterations = 100, initTemp = 3000, init_c = 5, a = 0.97, cv=5):
    C_axis_values = params_grid['C']
    gamma_axis_values = params_grid['gamma']
    init_history(len(C_axis_values), len(gamma_axis_values))
    currentTrialSolution = starting_point # index in C_axis_values and gamma_axis_values
    currentTemp = initTemp
    currentCost = evaluate_function(currentTrialSolution, params_grid, x_train, y_train, cv)
    print("------- Initial solution cost: %.4f"  % currentCost)
    C = [0,0]
    for i in range(max_iterations):
        # Try to ascend from current solution a bit for 3 times.
        for j in range(3):
            C[0], C[1] = ascend(currentTrialSolution, x_train, y_train,params_grid, cv)
            #print('THIS IS THE COST AFTER ASCEND: %.4f' %C[1])
            currentTrialSolution = C[0]

        nextTrialSolution = get_next_candidate(init_c, initTemp, currentTemp, currentTrialSolution, params_grid)
        currentCost = evaluate_function(currentTrialSolution,params_grid, x_train, y_train, cv)
        nextCost = evaluate_function(nextTrialSolution,params_grid, x_train, y_train, cv)
        diff = abs(currentCost - nextCost)
        if nextCost >= currentCost:
            currentTrialSolution = nextTrialSolution
            currentCost = nextCost
        else:
            if random.uniform(0, 1) < math.exp(-diff/currentTemp):
                currentTrialSolution = nextTrialSolution
                currentCost = nextCost

        currentTemp = get_temperature(initTemp, i , a)
        print('Iteration %d. Next trial solution cost: %.4f' %(i, currentCost))
    print("-------- End of simulated annealing --------")
    print('Final cost: %.4f' % (currentCost))
    print("Total fits performed: %d" % (get_evals_counter()*cv))
    p = {'C': None, 'gamma': None, 'kernel': 'rbf'}
    p['C'] = params_grid['C'][currentTrialSolution[0]]
    p['gamma'] = params_grid['gamma'][currentTrialSolution[1]]
    print("The best params are: ")
    print(p)
    return p


class Simulated_annealing(BaseEstimator):
    def __init__(self, baseClassifier=SVC(),
                 parameters_grid={},
                 initial_trial_solution=[1,1],
                 cv=5,
                 max_iterations = 100,
                 initTemp = 3000,
                 init_c = 5,
                 # Temperature reduction factor
                 a = 0.97):
        self.baseClassifier = baseClassifier
        self.parameters_grid = parameters_grid
        self.initial_trial_solution = initial_trial_solution
        self.cv = cv
        self.max_iterations=max_iterations
        self.initTemp = initTemp
        self.init_c=init_c
        self.a = a
        return

    def get_params(self, deep=True):
        return {"baseClassifier": self.baseClassifier,
                "parameters_grid": self.parameters_grid,
                "initial_trial_solution": self.initial_trial_solution,
                "cv": self.cv,
                "max_iterations": self.max_iterations,
                "initTemp": self.initTemp,
                "init_c": self.init_c,
                "a": self.a
        }

    def set_params(self, **parameters):
        for parameter, value, in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        params = simulated_annealing(self.initial_trial_solution, X, y, self.parameters_grid,
                                     self.max_iterations, self.initTemp, self.init_c, self.a, self.cv)
        self.baseClassifier = self.baseClassifier.set_params(**params)
        self.baseClassifier.fit(X, y)
        return

    def predict(self, X):
        y_predictions = self.baseClassifier.predict(X)
        return y_predictions