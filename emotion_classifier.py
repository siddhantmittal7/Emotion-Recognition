import time
import argparse
import os
import sys
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0):
        import _pickle as cPickle
else:
        import cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from parameters import DATASET, TRAINING, HYPERPARAMS

def train(epochs=HYPERPARAMS.epochs, random_state=HYPERPARAMS.random_state, 
          kernel=HYPERPARAMS.kernel, decision_function=HYPERPARAMS.decision_function, gamma=HYPERPARAMS.gamma, train_model=True):

        print( "loading dataset " + DATASET.name + "...")
                
        data, validation, test = load_data(validation=True, test=True)
        
        print(data)
        
        if train_model:
            # Training phase
            print( "building model...")
            model = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function, gamma=gamma)

            print( "start training...")
            print( "--")
            print( "kernel: {}".format(kernel))
            print( "decision function: {} ".format(decision_function))
            print( "max epochs: {} ".format(epochs))
            print( "gamma: {} ".format(gamma))
            print( "--")
            print( "Training samples: {}".format(len(data['Y'])))
            print( "Validation samples: {}".format(len(validation['Y'])))
            print( "--")
            start_time = time.time()
            model.fit(data['X'], data['Y'])
            training_time = time.time() - start_time
            print( "training time = {0:.1f} sec".format(training_time))
            
#             value = 1.5
#             width = 0.75

#             plot_decision_regions(X=data['X'], 
#                       y=data['Y'],
#                       clf=model,filler_feature_values={2: value, 3: value, 4: value, 5: value},
#                       filler_feature_ranges={2: width, 3: width, 4: width, 5: width},
#                       res=0.02)

#             # Update plot object with X/Y axis labels and Figure Title
#             plt.xlabel(data.columns[0], size=14)
#             plt.ylabel(data.columns[1], size=14)
#             plt.title('SVM Decision Region Boundary', size=16)

            if TRAINING.save_model:
                print( "saving model...")
                with open(TRAINING.save_model_path, 'wb') as f:
                        cPickle.dump(model, f)

            print( "evaluating...")
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print( "  - validation accuracy = {0:.1f}".format(validation_accuracy*100))
            return validation_accuracy
        else:
            # Testing phase : load saved model and evaluate on test dataset
            print( "start evaluation...")
            print( "loading pretrained model...")
            if os.path.isfile(TRAINING.save_model_path):
                with open(TRAINING.save_model_path, 'rb') as f:
                        model = cPickle.load(f)
            else:
                print( "Error: file '{}' not found".format(TRAINING.save_model_path))
                exit()

            print( "--")
            print( "Validation samples: {}".format(len(validation['Y'])))
            print( "Test samples: {}".format(len(test['Y'])))
            print( "--")
            print( "evaluating...")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'],  validation['Y'])
            print( "  - validation accuracy = {0:.1f}".format(validation_accuracy*100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print( "  - test accuracy = {0:.1f}".format(test_accuracy*100))
            print( "  - evalution time = {0:.1f} sec".format(time.time() - start_time))
            return test_accuracy

def evaluate(model, X, Y):
        predicted_Y = model.predict(X)
        accuracy = accuracy_score(Y, predicted_Y)
        
        
        # x1 = np.linspace(0, 1, 50)
        
        # l1= predicted_Y[:50]
        # l2 = Y[:50]
        
        # plt.plot(x1,l1,label='Predicted')
        # plt.plot(x1,l2,label='Actual')
        # plt.legend(loc='upper left')
        # plt.show()
        
        return accuracy

    
def load_data(validation=True, test=True):
    
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":
        print("here$$$$$$$$$$$$$$$")
        # load train set
        data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
        data_dict['X'] = np.array([x.flatten() for x in data_dict['X']])
        data_dict['X'] = np.concatenate((data_dict['X'], np.load(DATASET.train_folder + '/hog_features.npy')), axis=1)
        data_dict['X'] = np.concatenate((data_dict['X'], np.array([x.flatten() for x in np.load(DATASET.train_folder + '/gabor_features.npy')])), axis=1)
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to]
        if validation:
            # load validation set 
            validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
            validation_dict['X'] = np.array([x.flatten() for x in validation_dict['X']])
            validation_dict['X'] = np.concatenate((validation_dict['X'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)
            validation_dict['X'] = np.concatenate((validation_dict['X'], np.array([x.flatten() for x in np.load(DATASET.validation_folder + '/gabor_features.npy')])), axis=1)
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to]
        if test:
            # load train set
            test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
            test_dict['X'] = np.array([x.flatten() for x in test_dict['X']])
            test_dict['X'] = np.concatenate((test_dict['X'], np.load(DATASET.test_folder + '/hog_features.npy')), axis=1)
            test_dict['X'] = np.concatenate((test_dict['X'], np.array([x.flatten() for x in np.load(DATASET.test_folder + '/gabor_features.npy')])), axis=1)
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')
            np.save(DATASET.test_folder + "/lab.npy", test_dict['Y'])
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to]

        return data_dict, validation_dict, test_dict
    else:
        print( "Unknown dataset")
        exit()

print("start")
train()
print("all done")