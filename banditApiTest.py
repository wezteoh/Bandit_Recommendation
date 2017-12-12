'''
Example of API usage on how to work with this library
Tests almost everything.
'''
from sclrecommender.mask import MaskGenerator
from sclrecommender.mask import RandomMaskGenerator
from sclrecommender.mask import LegalMoveMaskGenerator

from sclrecommender.matrix import RatingMatrix
from sclrecommender.matrix import PositiveNegativeMatrix
from sclrecommender.matrix import OneClassMatrix

from sclrecommender.analyzer import MatrixAnalyzer

from sclrecommender.transform import MatrixTransform

from sclrecommender.parser import ExampleParser
from sclrecommender.parser import MovieLensParser100k

# from sclrecommender.bandit.runner import BanditRunner
from banditRunner2541 import BanditRunner2541 

#from sclrecommender.bandit.model import UncertaintyModel
from uncertaintyModel import UncertaintyModel
#from nnmf import NNMF # Distribution SVI NNMF
#from nnmf_vanilla import NNMFVanilla # NNMFVanilla with point estimates

from pmf import PMF

from banditChoiceBoltzmann import BanditChoiceBoltzmann
from banditChoiceUCBEmpirical import BanditChoiceUCBEmpirical
from banditChoiceExploit import BanditChoiceExploit
from banditChoiceThompsonSampling import BanditChoiceThompsonSampling
from banditChoiceEntropy import BanditChoiceEntropy
from banditChoiceEgreedy import BanditChoiceEgreedy

from sclrecommender.bandit.choice import RandomChoice # A random choice
from sclrecommender.bandit.choice import OptimalChoice # The optimal choice
from sclrecommender.bandit.choice import WorstChoice # The worst choice

from utils import preprocess_data, prepare_test_users

from sclrecommender.evaluator import Evaluator
# Reconstruction Evaluators
from sclrecommender.evaluator import ReconstructionEvaluator
from sclrecommender.evaluator import RootMeanSquareError
from sclrecommender.evaluator import PositiveNegativeEvaluator
from sclrecommender.evaluator import F1ScoreEvaluator

# Ranking Evaluators
from sclrecommender.evaluator import RankingEvaluator
from sclrecommender.evaluator import RecallAtK
from sclrecommender.evaluator import PrecisionAtK
from sclrecommender.evaluator import MeanAveragePrecisionAtK
# TODO: NDCG, used by 2017 paper by Pierre

# Bandit Evaluators 
from sclrecommender.evaluator import RegretOptimalEvaluator
from sclrecommender.evaluator import RegretInstantaneousEvaluator

import copy 
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

'''
def pprint(obj):
    # For debugging, print statements with numpy variable names and shape
    def namestr(obj):
        namespace = globals()
        return [name for name in namespace if namespace[name] is obj]
    # Assumes obj is a numpy array, matrix
    try:
        print(namestr(obj), obj.shape)
    except:
        try:
            print(namestr(obj), ",", len(obj))
        except:
            print(namestr(obj))
    print(obj)
'''
def pprint(obj):
    print(obj)

def runAll(nnmf, ucb, ratingMatrix, trainMatrix, testMatrix, testUsers, modelName, fileLocation):
    positiveThreshold = 3.0 # Threshold to set prediction to positive labels
    labelTruth = PositiveNegativeMatrix(ratingMatrix, positiveThreshold)

    positiveNegativeMatrix = labelTruth.getPositiveNegativeMatrix()

    pprint(trainMatrix)
    pprint(testMatrix)
    pprint(positiveNegativeMatrix)


    # Step 5: RecommenderAlgorithm
    # Option 5.1: ReconstructionMatrix: Outputs a reconstruction of actual matrix, known as recommenderMatrix

    reconstructionMatrix = ratingMatrix.copy() # TODO: Calculate reconstruction matrix 
    reconstructionPrediction = PositiveNegativeMatrix(reconstructionMatrix, positiveThreshold)
    positiveNegativePredictionMatrix = reconstructionPrediction.getPositiveNegativeMatrix()

    pprint(reconstructionMatrix)
    pprint(positiveNegativePredictionMatrix)

    # Option 5.2: RankingMatrix: Outputs a matrix of ranking for each user or item
    # Bandit Specific, get Legal Move that can be trained on
    legalTrainMask = LegalMoveMaskGenerator(trainMatrix).getMaskCopy()
    legalTestMask = LegalMoveMaskGenerator(testMatrix).getMaskCopy()

    pprint(legalTrainMask)
    pprint(legalTestMask)

    banditRunner = BanditRunner2541(ratingMatrix.copy(), legalTrainMask.copy(), legalTestMask.copy(), testUsers, modelName, fileLocation)
    banditRunner.setUncertaintyModel(nnmf)
    banditRunner.setBanditChoice(ucb)
    
    rankingMatrix = banditRunner.generateRanking()
    orderChoices = banditRunner.getOrderChoices()

    pprint(rankingMatrix)
    pprint(orderChoices)

    # Step 6: Evaluator

    # Option 6.1 Reconstruction Matrix evaluators
    accuracy = ReconstructionEvaluator(ratingMatrix, reconstructionMatrix).evaluate()
    rmse = RootMeanSquareError(ratingMatrix, reconstructionMatrix).evaluate()
    f1ScoreEvaluator = F1ScoreEvaluator(ratingMatrix, reconstructionMatrix)
    f1Score = f1ScoreEvaluator.evaluate()
    recall = f1ScoreEvaluator.getRecall()
    precision= f1ScoreEvaluator.getPrecision()

    pprint(accuracy)
    pprint(rmse)
    pprint(f1Score)
    pprint(recall)
    pprint(precision)

    # Option 6.2  Ranking Matrix evaluators
    # Option 6.2.1 Confusion Matrix evaluators
    # Evaluate the ranking matrix that was given
    k = 10 # number of items for each user is 20, so should be less than 20 so recall not guaranteed to be 1

    #-----------------------------------------------------------------------
    print("TEMP SHRINK TO tempMaxNumUser!")
    print(ratingMatrix.shape)
    ratingMatrix = ratingMatrix[:tempMaxNumUser]
    # Choices were made from any position, so can't reduce rating matrix size by tempMaxNumItem
    # ratingMatrix = ratingMatrix[:, :tempMaxNumItem]
    print(ratingMatrix.shape)
    print(legalTestMask.shape)
    legalTestMask = legalTestMask[:tempMaxNumUser]
    # legalTestMask = legalTestMask[:, :tempMaxNumItem]
    print(legalTestMask.shape)
    print(rankingMatrix.shape)
    rankingMatrix = rankingMatrix[:tempMaxNumUser]
    # rankingMatrix = rankingMatrix[:, :tempMaxNumItem]
    print(rankingMatrix.shape)
    #-------------------------------------------------------------------------------------------------------
    meanPrecisionAtK = PrecisionAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    meanRecallAtK = RecallAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    meanAveragePrecisionAtK = MeanAveragePrecisionAtK(ratingMatrix, rankingMatrix, positiveThreshold, k).evaluate()
    print("Model: " + str(modelName))
    print("\nMeanRecallAtK")
    print(meanRecallAtK) # meanRecallAtK 
    print("MeanPrecisionAtK")
    print(meanPrecisionAtK) # meanPrecisionAtK 
    print("MeanAveragePrecisionAtK")
    print(meanAveragePrecisionAtK)

    # Option 6.2.2  Bandit evaluators 
    discountFactor = 0.99
    regretBasedOnOptimalRegret = RegretOptimalEvaluator(ratingMatrix, rankingMatrix, discountFactor).evaluate()
    instantaneousRegret = RegretInstantaneousEvaluator(ratingMatrix, rankingMatrix, discountFactor, legalTestMask, orderChoices)
    regretBasedOnInstantaneousRegret = instantaneousRegret.evaluate()
    cumulativeInstantaneousRegret =  instantaneousRegret.getCumulativeInstantaneousRegret()
    print("RegretBasedOnOptimalRegret")
    pprint(regretBasedOnOptimalRegret)
    print("RegretBasedOnInstantaneousRegret")
    pprint(regretBasedOnInstantaneousRegret)
    print("CumulativeInstantaneousRegret")
    pprint(cumulativeInstantaneousRegret)

    instantRegretPerUser = []
    cumulativeRegretPerUser = []
    for userIndex in range(tempMaxNumUser):
        regretBasedOnInstantaneousRegret = instantaneousRegret.evaluate(userIndex)
        cumulativeInstantaneousRegret =  instantaneousRegret.getCumulativeInstantaneousRegret()

        instantRegretPerUser.append(regretBasedOnInstantaneousRegret)
        cumulativeRegretPerUser.append(cumulativeInstantaneousRegret)
    meanInstantRegret = np.mean(np.array(instantRegretPerUser))
    print("MeanInstantRegret")
    print(meanInstantRegret)
    # cumRegretUser = np.array(cumulativeRegretPerUser)
    cumRegretUser = np.array([np.array(xi) for xi in cumulativeRegretPerUser])

    print("CumulativeInstantRegretPerUser")
    print(cumRegretUser.shape)
    print(cumRegretUser)
    meanCumRegretUser = np.mean(cumRegretUser, axis = 0)
    print("MeanCumulativeInstantRegretPerUser")
    print(meanCumRegretUser.shape)
    print(meanCumRegretUser)
    print("Ranking matrix for numUser users and numItems items")
    print(rankingMatrix[:, :tempMaxNumItem])

    #-----------------------------------------------------------------
    matrixAnalyzer.summarize() 
    xs = []
    ys = []
    for i in range(cumRegretUser.shape[0]):
        x = list(range(len(cumRegretUser[i])))
        y = cumRegretUser[i].copy()
        xs.append(x)
        ys.append(y)
    #-----------------------------------------------------------------
    return xs, ys

    #-------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # seedNum =  int(random.random() * 1000)
    # print("SEEDNUM IS", seedNum)
    seedNum = 196
    np.random.seed(seedNum)
    random.seed(seedNum)

    
    # Anything with pprint(numpyVariable) means it is a numpy matrix
    # Step 1: Get data based on dataset specific parser
    # dataDirectory = "sclrecommender/data/movielens/ml-100k"
    dataDirectory ="ml-100k"
    mlp = MovieLensParser100k(dataDirectory)
    numUser = 50 
    numItem = 50
    exParser = ExampleParser("")
    ratingMatrix = exParser.getRatingMatrix(numUser, numItem)
    ratingMatrix[0][0] = 1.0
    ratingMatrix = mlp.getRatingMatrixCopy()

    R = ratingMatrix.copy()

    # To run on entire rating matrix
    ratingMatrix = mlp.getRatingMatrixCopy()
    # Remove the users that are too hot
    ratingMatrix = MatrixTransform(ratingMatrix).coldUsers(830)
    # Sort by users that are hot
    ratingMatrix = MatrixTransform(ratingMatrix).hotUsers()

    # TODO: Work with R
    tempMaxNumUser = 20 # TODO TEMPORARY, FOLLOWS NUMBER IN BANDIT RUNNER
    tempMaxNumItem = 50 # for printing ranking matrix
    print("Test Users")
    R = preprocess_data(R)
    test_users = prepare_test_users(R)
    print(test_users)

    fileLocation = "/home/soon/Desktop/runs/all"
    # '''
    fileLocation = "/home/soon/Desktop/runs/dense"
    # Dense
    ratingMatrix = R.copy()
    for curr in range(test_users.shape[0]):
        # Swap them
        userIndex = test_users[curr]
        temp = ratingMatrix[curr].copy()
        temp2 = ratingMatrix[userIndex].copy()
        ratingMatrix[curr] = temp2.copy()
        ratingMatrix[userIndex] = temp.copy()
    # '''

    testUsers = np.array([i for i in range(tempMaxNumUser)])

    # TODO: PICK FOR DENSE AND ALL
    # Create the folder
    bashCommand = "mkdir -p " + fileLocation
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


    # Step 2: Generate both Rating Matrix and Label Matrix for evaluation
    rmTruth = RatingMatrix(ratingMatrix)
    # Step 3 Analyze the rating matrix
    matrixAnalyzer = MatrixAnalyzer(ratingMatrix)
    matrixAnalyzer.summarize()

    # Step 4: Split rating matrix to train and test
    trainSplit = 0.70 # Assuming 200 items, 70 percent, will allow > 50 items to be explored in test

    # Step 4.1: Choose splitting procedure

    # Option 4.1.1: Random Split
    randomMaskTrain, randomMaskTest = RandomMaskGenerator(rmTruth.getRatingMatrix(), trainSplit).getMasksCopy()

    #TODO: Option 4.1.2: Split based on time
    #TODO: Option 4.1.3: Split based on cold users
    #TODO: Option 4.1.4: Split based on cold items
   
    # Step 4.2: Apply mask
    rmTrain = copy.deepcopy(rmTruth)
    rmTest = copy.deepcopy(rmTruth)
    rmTrain.applyMask(randomMaskTrain)
    rmTest.applyMask(randomMaskTest)

    trainMatrix = rmTrain.getRatingMatrix()
    testMatrix = rmTest.getRatingMatrix()

    xLabel = 'Exploration Number'
    yLabel = 'Cumulative Instantaneous Regret'
    #----------------------------------------
    um = UncertaintyModel(ratingMatrix.copy())
    worstChoice = WorstChoice()
    modelString8 = "Worst"
    x8s, y8s = runAll(um, worstChoice, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString8, fileLocation)
    currI = 0
    for x8, y8 in zip(x8s, y8s):
        plt.plot(x8, y8, label=modelString8 + str(currI))
        currI += 1
    x8 = x8s[0]
    y8 = np.mean(y8s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString8)
    plt.savefig(fileLocation + "worstChoices.png")
    plt.clf()
    plt.plot(x8, y8, label=modelString8 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString8)
    plt.savefig(fileLocation + "worstChoicesMean.png")
    np.save(fileLocation + "x8.npy", x8)
    np.save(fileLocation + "y8s.npy", y8s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    bBoltz = BanditChoiceBoltzmann()
    modelString1 = "PMF_Boltzmann"
    x1s, y1s = runAll(pmf, bBoltz, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString1, fileLocation)
    currI = 0
    for x1, y1 in zip(x1s, y1s):
        plt.plot(x1, y1, label=modelString1 + str(currI))
        currI += 1
    x1 = x1s[0]
    y1 = np.mean(y1s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString1)
    plt.savefig(fileLocation + modelString1 + ".png")
    plt.clf()
    plt.plot(x1, y1, label=modelString1 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString1)
    plt.savefig(fileLocation + modelString1 + "Mean.png")
    np.save(fileLocation + "x1.npy", x1)
    np.save(fileLocation + "y1s.npy", y1s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    bExploit = BanditChoiceExploit()
    modelString2 = "PMF_Exploit"
    x2s, y2s = runAll(pmf, bExploit, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(),testUsers, modelString2, fileLocation)
    currI = 0
    for x2, y2 in zip(x2s, y2s):
        plt.plot(x2, y2, label=modelString2 + str(currI))
        currI += 1
    x2 = x2s[0]
    y2 = np.mean(y2s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString2)
    plt.savefig(fileLocation + modelString2 + ".png")
    plt.clf()
    plt.plot(x2, y2, label=modelString2 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString2)
    plt.savefig(fileLocation + modelString2 + "Mean.png")
    np.save(fileLocation + "x2.npy", x2)
    np.save(fileLocation + "y2s.npy", y2s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    bUcbEmpirical = BanditChoiceUCBEmpirical()
    modelString3 = "PMF_UCB"
    x3s, y3s = runAll(pmf, bUcbEmpirical, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(),testUsers, modelString3, fileLocation)
    currI = 0
    for x3, y3 in zip(x3s, y3s):
        plt.plot(x3, y3, label=modelString3 + str(currI))
        currI += 1
    x3 = x3s[0]
    y3 = np.mean(y3s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString3)
    plt.savefig(fileLocation + modelString3 + ".png")
    plt.clf()
    plt.plot(x3, y3, label=modelString3 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString3)
    plt.savefig(fileLocation + modelString3 + "Mean.png")
    np.save(fileLocation + "x3.npy", x3)
    np.save(fileLocation + "y3s.npy", y3s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    bThompson = BanditChoiceThompsonSampling()
    modelString4 = "PMF_Thompson_Sampling"
    x4s, y4s = runAll(pmf, bThompson, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString4, fileLocation)
    currI = 0
    for x4, y4 in zip(x4s, y4s):
        plt.plot(x4, y4, label=modelString4 + str(currI))
        currI += 1
    x4 = x4s[0]
    y4 = np.mean(y4s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString4)
    plt.savefig(fileLocation + modelString4 + ".png")
    plt.clf()
    plt.plot(x4, y4, label=modelString4 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString4)
    plt.savefig(fileLocation + modelString4 + "Mean.png")
    np.save(fileLocation + "x4.npy", x4)
    np.save(fileLocation + "y4s.npy", y4s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    bEntropy = BanditChoiceEntropy()
    modelString5 = "PMF_Entropy"
    x5s, y5s = runAll(pmf, bEntropy, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString5, fileLocation)
    currI = 0
    for x5, y5 in zip(x5s, y5s):
        plt.plot(x5, y5, label=modelString5 + str(currI))
        currI += 1
    x5 = x5s[0]
    y5 = np.mean(y5s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString5)
    plt.savefig(fileLocation + modelString5 + ".png")
    plt.clf()
    plt.plot(x5, y5, label=modelString5 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString5)
    plt.savefig(fileLocation + modelString5 + "Mean.png")
    np.save(fileLocation + "x5.npy", x5)
    np.save(fileLocation + "y5s.npy", y5s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    bEgreedy= BanditChoiceEgreedy()
    modelString6 = "PMF_eGreedy"
    x6s, y6s = runAll(pmf, bEgreedy, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString6, fileLocation)
    currI = 0
    for x6, y6 in zip(x6s, y6s):
        plt.plot(x6, y6, label=modelString6 + str(currI))
        currI += 1
    x6 = x6s[0]
    y6 = np.mean(y6s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel) 
    plt.ylabel(yLabel)
    plt.title(modelString6)
    plt.savefig(fileLocation + modelString6 + ".png")
    plt.clf()
    plt.plot(x6, y6, label=modelString6 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString6)
    plt.savefig(fileLocation + modelString6 + "Mean.png")
    np.save(fileLocation + "x6.npy", x6)
    np.save(fileLocation + "y6s.npy", y6s)
    plt.clf()
    #----------------------------------------
    um = UncertaintyModel(ratingMatrix.copy())
    optimalChoice = OptimalChoice()
    modelString7 = "Optimal"
    x7s, y7s = runAll(um, optimalChoice, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString7, fileLocation)
    currI = 0
    for x7, y7 in zip(x7s, y7s):
        plt.plot(x7, y7, label=modelString7 + str(currI))
        currI += 1
    x7 = x7s[0]
    y7 = np.mean(y7s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString7)
    plt.savefig(fileLocation + "optimalChoices.png")
    plt.clf()
    plt.plot(x7, y7, label=modelString7 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString7)
    plt.savefig(fileLocation + "optimalChoicesMean.png")
    np.save(fileLocation + "x7.npy", x7)
    np.save(fileLocation + "y7s.npy", y7s)
    plt.clf()
    #----------------------------------------
    pmf = PMF(ratingMatrix.copy())
    rc = RandomChoice()
    modelString9 = "PMF_Random"
    x9s, y9s = runAll(pmf, rc, ratingMatrix.copy(), trainMatrix.copy(), testMatrix.copy(), testUsers, modelString9, fileLocation)
    currI = 0
    for x9, y9 in zip(x9s, y9s):
        plt.plot(x9, y9, label=modelString9 + str(currI))
        currI += 1
    x9 = x9s[0]
    y9 = np.mean(y9s, axis = 0)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString9)
    plt.savefig(fileLocation + "randomChoices.png")
    plt.clf()
    plt.plot(x9, y9, label=modelString9 + str(currI))
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString9)
    plt.savefig(fileLocation + "randomChoicesMean.png")
    np.save(fileLocation + "x9.npy", x9)
    np.save(fileLocation + "y9s.npy", y9s)
    plt.clf()
    #----------------------------------------
    modelString = "All Models"
    plt.plot(x1, y1, label=modelString1)
    plt.plot(x2, y2, label=modelString2)
    plt.plot(x3, y3, label=modelString3)
    plt.plot(x4, y4, label=modelString4)
    plt.plot(x5, y5, label=modelString5)
    plt.plot(x6, y6, label=modelString6)
    plt.plot(x7, y7, label=modelString7)
    plt.plot(x8, y8, label=modelString8)
    plt.plot(x9, y9, label=modelString9)
    plt.legend(loc = 'upper left')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(modelString)
    plt.savefig(fileLocation + "AllInOne.png")
    plt.clf()
    print("DONE TESTING")
