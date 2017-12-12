import copy
import numpy as np

class BanditRunner2541(object):
    '''
    A bandit runner
    '''
    #--------------------------------------------------------------------------------------
    # Public Methods
    #--------------------------------------------------------------------------------------

    def __init__(self, ratingMatrix, legalTrainMask, legalTestMask, modelName=""):
        self.ratingMatrix = ratingMatrix
        self.legalTrainMask = legalTrainMask
        self.legalExploreMask = legalTestMask
        self.uncertaintyModel = None
        self.banditChoice = None
        # Update these hard-coded values
        # TODO: mkdir for these hard-coded values
        self.fileLocation = "/home/soon/Desktop/models/"
        self.modelName = modelName
        self.fileExt = ".bin"
        self.seedNumber = 521
        self.numSamples = 100
        self.rankingMatrix = None
        self.orderChoices = None

    def setUncertaintyModel(self, uncertaintyModel):
        self.uncertaintyModel = uncertaintyModel

    def setBanditChoice(self, banditChoice):
        self.banditChoice = banditChoice

    def generateRanking(self):
        rankingMatrix = self.run().copy()
        return rankingMatrix

    def getOrderChoices(self):
        if self.orderChoices is None:
            raise Exception("Must call generateRanking() first!")
        return self.orderChoices

    #--------------------------------------------------------------------------------------
    # Private methods
    #--------------------------------------------------------------------------------------
    def reset(self):
        self.uncertaintyModel.reset(self.seedNumber)
        trainMask = self.legalTrainMask.copy()
        exploreMask = self.legalExploreMask.copy()
        # To store results 
        self.rankingMatrix = np.zeros(exploreMask.shape)
        self.orderChoices = []
        # Train once
        self.uncertaintyModel.train(trainMask, None, True)
        self.uncertaintyModel.save(self.fileLocation + self.modelName + self.fileExt)

    def getMaxExplorationNumberUser(self, oneHotExplorationMask):
        # Assume explorationMask is 1 hot. 
        # 0 => Can't explore
        # 1 => Can explore
        if not np.array_equal(oneHotExplorationMask, oneHotExplorationMask.astype(bool)):
            raise ValueError("oneHotExplorationMask is Not One hot")
        return np.sum(oneHotExplorationMask, axis=1)

    def getMaxExplorationNumber(self, oneHotExplorationMask):
        # Assume explorationMask is 1 hot. 
        # 0 => Can't explore
        # 1 => Can explore
        if not np.array_equal(oneHotExplorationMask, oneHotExplorationMask.astype(bool)):
            raise ValueError("oneHotExplorationMask is Not One hot")
        return np.sum(oneHotExplorationMask) 

    def run(self):
        self.reset()
        trainMask = self.legalTrainMask.copy()
        exploreMask = self.legalExploreMask.copy()
        # Start from trained model
        maxStateCounter = self.getMaxExplorationNumberUser(exploreMask)
        
        # TODO: REMOVE ALL THE TEMP BELOW THAT WAS ONLY TEMPORARY
        tempMaxNumUser = 50 # Try out on the first 50 users
        tempMaxNumItem = 30 # To try out users with 30 items

        # Iterate for each user separately
        for userIndex in range(exploreMask.shape[0]):
            if userIndex >= tempMaxNumUser:
                # Return only the ranking matrix for those users
                return self.rankingMatrix[:tempMaxNumUser]
            print("ChoiceUser: ", userIndex)
            # TODO: Need some way of copying or initializing, not sure if this works, it works for now since loading
            currUncertaintyModel = self.uncertaintyModel
            # Load from scratch for eachuser
            currUncertaintyModel.load(self.fileLocation + self.modelName + self.fileExt)
            # Run main iteration loop
            for stateCounter in range(int(maxStateCounter[int(userIndex)])):
                if stateCounter >= tempMaxNumItem:
                    break
                currUncertaintyModel.load(self.fileLocation + self.modelName + self.fileExt)
                # Train but not fully
                currUncertaintyModel.train(trainMask, userIndex, False)
                modelSpecific = "_user_" + str(int(userIndex)) + "_item_" + str(int(stateCounter)) 
                currUncertaintyModel.save(self.fileLocation + self.modelName + modelSpecific + self.fileExt)
                #posterior = currUncertaintyModel.sampleForUser(userIndex, self.numSamples)
                posterior = currUncertaintyModel.sample_for_user(userIndex, self.numSamples)
                choiceItem = int(round(self.banditChoice.evaluate(posterior, exploreMask[userIndex].copy(), self.ratingMatrix[userIndex].copy())))
                # Processing of choice and saving relevant evaluation metadata
                print("StateCounter: ", stateCounter)
                print("ChoiceItem: ", choiceItem)
                trainMask, exploreMask = self.explore(userIndex, choiceItem, trainMask, exploreMask, stateCounter)

    
        self.uncertaintyModel.save_uncertainty_progress("", self.modelName, folder='BanditProgress')
        return self.rankingMatrix

    def explore(self, choiceUser, choiceItem, trainMask, exploreMask, stateCounter):
        # TODO: REMOVE TEMP DEBUG STATEMENTS BELOW
        print("Current user: ", choiceUser)
        print("Places for exploreMask for currentUser")
        print(np.where(exploreMask[choiceUser] == 1.0))
        print("Places for trainMask for currentUser")
        print(np.where(trainMask[choiceUser] == 0.0))

        if choiceUser >= exploreMask.shape[0] or choiceItem >= exploreMask.shape[1]:
            raise ValueError("Choice given is invalid in exploreMask")
        if choiceUser < 0 or choiceItem < 0:
            raise ValueError("Choice given is invalid in exploreMask")
        if exploreMask[choiceUser][choiceItem] == 0.0:
            raise ValueError("Choice given is invalid in exploreMask")
        if trainMask[choiceUser][choiceItem] == 1.0:
            raise ValueError("Choice given is invalid in trainMask")
        # Can now train on selected mask, set to 1.0
        trainMask[choiceUser][choiceItem] = 1.0
        # Can no longer explore selected mask, set to 0.0
        exploreMask[choiceUser][choiceItem] = 0.0
        # Update ranking matrix
        self.rankingMatrix[choiceUser][stateCounter] = self.ratingMatrix[choiceUser][choiceItem]
        self.orderChoices.append((choiceUser, choiceItem))
        return trainMask, exploreMask
