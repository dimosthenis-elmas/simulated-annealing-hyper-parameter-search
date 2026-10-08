import pandas as pd
from keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from simulated_annealing_module import simulated_annealing as sa
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""By flatten, we mean converting all images to vectors (i.e. of shape : (1,(nxn)) where n is the number of pixels in width or height)) 
So in the end the dataset will be of shape 
"""
def flatten_images(dataset):
    return dataset.reshape(dataset.shape[0], -1)

def plot_heatmap_2d(res, x_key, y_key, z_key, x_label, y_label, z_label):
    x = np.array(res[x_key].data)
    y = np.array(res[y_key].data)
    z = np.array(res[z_key])

    df = pd.DataFrame.from_dict(np.array([x, y, z]).T)
    df.columns = [x_label, y_label, z_label]
    df[z_label] = pd.to_numeric(df[z_label])
    pivoted = df.pivot(y_label, x_label, z_label)
    hm = sns.heatmap(pivoted, cmap='RdBu', annot=True, fmt=".4f", cbar_kws={'label': z_label})
    return hm

def get_train_test_sets(max_number_of_instances_for_training_set=None,
                              max_number_of_instances_for_test_set=None):
    print("Loading the dataset...")
    # Get a subset of the entire dataset. This is done in order to avoid long training times.
    # We will only train our model with this subset for this exercise.

    if(max_number_of_instances_for_training_set is None):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (X_Train_Entire, Y_Train_Entire), (X_Test_Entire, Y_Test_Entire) = mnist.load_data()
        # Note that this is a stratified split. It preserves the percentages lf each class being the same as in the original dataset.
        x_train, _, y_train, _, = train_test_split(X_Train_Entire,
                                                   Y_Train_Entire,
                                                   train_size=max_number_of_instances_for_training_set,
                                                   random_state=42,
                                                   stratify=Y_Train_Entire)

        _, x_test, _, y_test = train_test_split(X_Test_Entire,
                                                Y_Test_Entire,
                                                test_size=max_number_of_instances_for_test_set,
                                                random_state=42,
                                                stratify=Y_Test_Entire)

    x_train = flatten_images(x_train)
    x_test = flatten_images(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    return x_train, y_train, x_test, y_test


print("Loading the MNIST dataset...")
x_train, y_train, x_test, y_test = get_train_test_sets(100, 25)

print("Total number of fits for entire grid search is going to be: %d" % (5 * 64))
print("Let's do the entire grid search and plot a heatmap")

scoring = 'f1_macro'
params = {'C': [1e-3, 0.01, 0.1, 1, 2, 3, 4, 5,6,7,8,10,12,15,20,30,40,50,60,70,80,90,100],'gamma': [16,15,14,13,12,10,9,8,7,6,5,4,3, 2,1, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], 'kernel':['rbf']}

clf = GridSearchCV(SVC(), params, scoring=[scoring], refit=scoring, verbose=1, cv=5, n_jobs=-1)
clf.fit(x_train, y_train)
hm = plot_heatmap_2d(res=clf.cv_results_,
                x_key='param_C', y_key='param_gamma', z_key='mean_test_'+scoring,
                x_label='C', y_label='gamma', z_label=scoring)

#plt.show()

# Simulated annealing for SVC with RBF kernel

print("Next, let's try using simulated annealing for the same grid search:")

clf= sa.Simulated_annealing(
    baseClassifier=SVC(kernel='rbf'),
    parameters_grid = {'C': [1e-3, 0.01, 0.1, 1, 2, 3, 4, 5,6,7,8,10,12,15,20,30,40,50,60,70,80,90,100],'gamma': [16,15,14,13,12,10,9,8,7,6,5,4,3, 2,1, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
    # The initial trial solution for starting the algorithm.
    # Note that these are the coordinates in the grid of hyper parameters.
    # In this example it would be C = 0.1 and gamma = 1
    initial_trial_solution=[1,1],
    # Cross validation folds. In each iteration the algorithm performs a cross validated fit.
    cv= 5,
    max_iterations = 100,
    # Initial temperature. The higher the temperature, the greatest the 'turbulence' in the trial solutions hyper params.
    initTemp = 2500,
    # Initial value for the disturbance factor. This controls the amount of the flunctuations of the hyper params on a given temperature.
    init_c = 8,
    # This factor is the temperature reduction factor. It controls the rate of cooling. Max is 1 (no temperature reduction at all)
    a = 0.97).fit(X =x_train, y=y_train)

h = sa.get_SA_history()

print(h.shape)
h = np.rot90(h.transpose((1, 0)),-1)
for j in range(h.shape[0]):
    for i in range(h.shape[1]):
        ax = hm
        if h[i][j] > 0:
            ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='white', lw=2))
            #ax.text(i, j, f'{h[i][j]:.4f}', size=9, color='purple')
        #print((i,j))
        #print(h[i][j])




plt.show()

