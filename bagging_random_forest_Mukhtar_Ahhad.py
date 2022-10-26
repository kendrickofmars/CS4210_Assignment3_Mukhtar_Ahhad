# -------------------------------------------------------------------------
# AUTHOR: Ahhad Mukhtar
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: 8 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = []  # this array will be used to count the votes of each classifier

# reading the training data from a csv file and populate dbTraining
# --> add your Python code here
with open(r"C:\Users\fourf\Documents\SchoolDocuments\CS4210 Notes & HW\Assignment3_Mukhtar_Ahhad\optdigits.tra",
          'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        dbTraining.append(row)
        print(row)
# reading the test data from a csv file and populate dbTest
# --> add your Python code here
with open(r"C:\Users\fourf\Documents\SchoolDocuments\CS4210 Notes & HW\Assignment3_Mukhtar_Ahhad\optdigits.tes",
          'r') as csvfile2:
    reader = csv.reader(csvfile2)
    for i, row in enumerate(reader):
        dbTest.append(row)
        print(row)
        classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
# --> add your Python code here

print("Started my base and ensemble classifier ...")

for k in range(
        20):  # we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

    bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

    # populate the values of X_training and y_training by using the bootstrapSample
    # --> add your Python code here
    for i in bootstrapSample:
        X_training.append(i[0:-1])
        y_training.append(i[-1])
    # fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=None)  # we will use a single decision tree without pruning it
    clf = clf.fit(X_training, y_training)
    prediction_right_counter = 0
    prediction_wrong_counter = 0
    for i, testSample in enumerate(dbTest):

        # make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
        # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
        # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
        # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
        # this array will consolidate the votes of all classifier for all test samples
        # --> add your Python code here
        predict = int(clf.predict([testSample[0:-1]])[0])
        classVotes[i][predict] += 1

        if k == 0:  # for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
            # --> add your Python code here
            if predict == int(testSample[-1]): #last val in row
                prediction_right_counter += 1
            else:
                prediction_wrong_counter += 1
    if k == 0:  # for only the first base classifier, print its accuracy here
        # --> add your Python code here
        accuracy = prediction_right_counter / (prediction_right_counter+prediction_wrong_counter)
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print("My base classifier accuracy: " + str(accuracy))
        print("")
    prediction_right_counter_2 = 0
    prediction_wrong_counter_2 = 0
    # now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
    # --> add your Python code here
    for i in range(len(dbTest)):
        if classVotes[i].index(max(classVotes[i])) == int(dbTest[i][-1]):
            prediction_right_counter_2 += 1
        else:
            prediction_wrong_counter_2 += 1
    accuracy_2 = prediction_right_counter_2 / (prediction_right_counter_2+prediction_wrong_counter_2)
    # printing the ensemble accuracy here
    print("Finished my ensemble classifier (slow but higher accuracy) ...")
    print("My ensemble accuracy: " + str(accuracy_2))
    print("")

    print("Started Random Forest algorithm ...")

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=20)  # this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

    # Fit Random Forest to the training data
    clf.fit(X_training, y_training)

    # make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
    # --> add your Python code here
    prediction_right_counter_3 = 0
    prediction_wrong_counter_3 = 0
    for i in dbTest:
        class_predicted_rf = int(clf.predict([testSample[:-1]])[0])
        if class_predicted_rf == int(testSample[-1]):
            prediction_right_counter_3 += 1
        else:
            prediction_wrong_counter_3 += 1
    accuracy_3 = prediction_right_counter_3 / (prediction_right_counter_3+prediction_wrong_counter_3)
    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here

    # printing Random Forest accuracy here
    print("Random Forest accuracy: " + str(accuracy_3))

    print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
