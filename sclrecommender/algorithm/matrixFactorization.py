from .recommenderMatrixAlgorithm import RecommenderMatrixAlgorithm

import numpy as np
import sys

class MatrixFactorization(RecommenderMatrixAlgorithm):
    def __init__(self, ratingMatrix):
        RecommenderMatrixAlgorithm.__init__(self, ratingMatrix)

    def executeRecommender(self, hiddenDimension, regConst=0.001, learningRate=0.00001, numIteration=1000):
    #def executeRecommender(self, hiddenDimension, regConst=0.001, learningRate=0.0001, numIteration=10000):
        self.recommenderMatrix = self.ratingMatrix.copy()

        # TODO: REDO WITHOUT TENSORFLOW
        import tensorflow as tf
    
        mask_ = np.ones(self.ratingMatrix.shape)
        # Need set mask for rating
        # mask_[np.where(self.ratingMatrix == 0)] = 0.0
        weight_ = np.ones(self.ratingMatrix.shape)

        R_ = self.ratingMatrix.copy()
        freqI = np.sum(self.ratingMatrix, axis=0)
        maxFreqI = np.max(freqI)
        temp = np.ones(freqI.shape)
        recFreqI = np.divide(temp, freqI, out=np.zeros_like(temp), where=freqI!=0.0)

        '''
        for userIndex in range(self.ratingMatrix.shape[0]):
            for itemIndex in range(self.ratingMatrix.shape[1]):
                # Serendipity
                if self.ratingMatrix[userIndex][itemIndex] == 1.0:
                    weight_[userIndex][itemIndex] = float(maxFreqI)/float(freqI[itemIndex])
                    continue
                # Censoring
                else:
                    weight_[userIndex][itemIndex] = recFreqI[itemIndex]
                    continue
        '''

        mask_ = weight_
        
        n_users, n_items = R_.shape
        latent_dim = hiddenDimension
        reg_rate = regConst
        learning_rate = learningRate

        R = tf.placeholder(tf.float32)

        # TODO: Change to gaussians
        U = tf.Variable(tf.random_normal([n_users, latent_dim], 0.0, 1.0), name='user_matrix')
        V = tf.Variable(tf.random_normal([n_items, latent_dim], 0.0, 1.0), name='item_matrix')

        # Mask could be for not trainign the zeros
        # OR could simply be the weighting of each term.
        mask = tf.placeholder(tf.float32)
        weight = tf.placeholder(tf.float64)
        feed_dict = {R: R_, mask: mask_, weight: weight_}

        reg = tf.norm(U) + tf.norm(V)
        #error = tf.reduce_sum(tf.multiply(mask, tf.multiply(weight_, tf.square(R - tf.matmul(U, V, transpose_b=True)))))
        error = tf.reduce_sum(tf.multiply(mask, tf.square(R - tf.matmul(U, V, transpose_b=True))))
        loss = error + reg_rate * reg

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
        initialize = tf.global_variables_initializer()

        losses = []
        R_hat = None
        with tf.Session() as sess:
            sess.run(initialize)
            for i in range(numIteration):
                sess.run(train, feed_dict)
                losses.append(sess.run(loss, feed_dict))

            R_hat = sess.run(tf.matmul(U, V, transpose_b=True), feed_dict)
        print("Original:")
        print(R_)
        print("Reconstruction:")
        print(R_hat)
        self.recommenderMatrix = R_hat

        # Multiply both by some mask to get each quartile's rmse
        # The must basically sets entire columns of items to zeros
        q1 = int(round(0.25 * maxFreqI))
        q2 = int(round(0.5 * maxFreqI))
        q3 = int(round(0.75 * maxFreqI))
        q4 = maxFreqI
        freqI = np.sum(self.ratingMatrix, axis=0)
        tileFreqI = np.tile(freqI, (R_.shape[0], 1))
        col1 = np.where(tileFreqI <= q1)
        col2 = np.where((tileFreqI <= q2) & (tileFreqI > q1))
        col3 = np.where((tileFreqI <= q3) & (tileFreqI > q2))
        col4 = np.where((tileFreqI <= q4) & (tileFreqI > q3))

        print("q1: ", q1)
        print("q2: ", q2)
        print("q3: ", q3)
        print("q4: ", q4)

        # TODO: FIXME
        quartileMask = np.ones(R_.shape)
        rmse = np.sqrt(np.mean(np.multiply(quartileMask, np.power((R_hat - R_),2.0))))
        # Individual RMSE do not sum up to final RMSE, something might be wrong!
        # MAYBE DONT HAVE TO SUM UP TO FINAL RMSE SINCE TAKING MEAN IS DIFFERENT BASED ON NUMBER OF VALUES
        print("Total RSME: ", rmse)

        tempDebug = np.zeros(R_.shape)
        tempDebug[col1] += 1.0
        tempDebug[col2] += 1.0
        tempDebug[col3] += 1.0
        tempDebug[col4] += 1.0
        quartileMask = tempDebug
        rmse = np.sqrt(np.mean(np.multiply(quartileMask, np.power((R_hat - R_),2.0))))
        print("RSME TEMP DEBUG: ", rmse)

        '''
        print("TEMP DEBUG: ", tempDebug)
        print(np.where(tempDebug == 1.0))
        print(np.where(tempDebug < 1.0))
        print(np.where(tempDebug > 1.0))
        Verified no intersections!
        '''

        quartileMask = np.zeros(R_.shape)
        quartileMask[col1] = 1.0
        rmse = np.sqrt(np.mean(np.multiply(quartileMask, np.power((R_hat - R_),2.0))))
        print("RSME Quartile 1: ", rmse)

        quartileMask = np.zeros(R_.shape)
        quartileMask[col2] = 1.0
        rmse = np.sqrt(np.mean(np.multiply(quartileMask, np.power((R_hat - R_),2.0))))
        print("RSME Quartile 2: ", rmse)

        quartileMask = np.zeros(R_.shape)
        quartileMask[col3] = 1.0
        rmse = np.sqrt(np.mean(np.multiply(quartileMask, np.power((R_hat - R_),2.0))))
        print("RSME Quartile 3: ", rmse)

        quartileMask = np.zeros(R_.shape)
        quartileMask[col4] = 1.0
        rmse = np.sqrt(np.mean(np.multiply(quartileMask, np.power((R_hat - R_),2.0))))
        print("RSME Quartile 4: ", rmse)

