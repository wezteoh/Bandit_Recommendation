from models.simple import SimpleMatrixFactorization as _PMF
from models.nnmf_svi_eddie import save_graph_parameters, load_graph_parameters
from uncertaintyModel import UncertaintyModel
import tensorflow as tf
import edward as ed
import _pickle
import os

class PMF(UncertaintyModel):
    """
    Thin wrapper around our model to conform to Soon's API.
    It also allows us to manage session/graph/hyperparams if we want.
    """

    def __init__(self, ratingMatrix, seed=None):
        self.ratingMatrix = ratingMatrix
        self.sess = None
        self.reset(seed=seed)
        self.user_mean_progress = {}
        self.user_var_progress = {}
        self.user_mse_progress = {}
        self.current_user = -1
        
    def reset(self, seed=None):
        tf.reset_default_graph()

        if seed is not None:
            ed.set_seed(seed) # sets seed for both tf and numpy

        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()

        with self.sess.as_default():
            #self.model = _PMF(
            #    self.ratingMatrix,
            #    hidden_dim=70, # To match NNMF 60 + 10 hidden dims.
            #    batch_size=200, n_samples=10, pR_stddev=1.)
            #self.model = _PMF(
            #    self.ratingMatrix,
            #    hidden_dim=20,
            #    batch_size=1000, n_samples=10, pR_stddev=1.)
            self.model = _PMF(
                self.ratingMatrix,
                hidden_dim=10,
                batch_size=200, n_samples=10, pR_stddev=2.,
                lr_init=0.01)
                
    def save(self, fname):
        with self.sess.as_default():
            return self.model.saver.save(self.sess, fname)

    def load(self, fname):
        with self.sess.as_default():
            self.model.saver.restore(self.sess, fname)

    def save_pkl(self, fname):
        with self.sess.as_default():
            save_graph_parameters(fname)
        return fname

    def load_pkl(self, fname):
        with self.sess.as_default():
            load_graph_parameters(fname)

    def train(self, legalTrainIndices, user, train_global):
        if train_global:
            n_iter = 2000
        
        else:
            n_iter = 500
        
        with self.sess.as_default():            
            losses = self.model.train(mask=legalTrainIndices, n_iter=n_iter)
            
            if not train_global:
                self.uncertainty_progress(user, legalTrainIndices, self.sess)
                
        return losses

    def sample_for_user(self, user_index, num_samples):
        # return (k, m) matrix of k samples for user i
        with self.sess.as_default():            
            return self.model.sample_user_ratings(user_index, num_samples)
            
    def uncertainty_progress(self, user, train_mask, sess):
        graph_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        U_means = graph_vars[0]
        U_vars  = graph_vars[1]
        V_means = graph_vars[2]
        
        u_mean = sess.run(U_means)[user]
        u_var = sess.run(U_vars)[user]
        
        avail = np.where((self.ratingMatrix[user]>0)-train_mask[user])[0]
        v_means = sess.run(V_means)[avail]
        rating_means = np.sum(u_mean*v_means, axis=1)
        true_ratings = R[user][avail]
        
        user_mse = np.mean(np.square(true_ratings-rating_means))
        
        
        if user != self.current_user:
            self.current_user = user
            self.user_mean_progress[user] = u_mean
            self.user_var_progress[user] = u_var
            self.user_mse_progress[user] = [user_mse]
        
        else: 
            self.user_mean_progress[user] = np.vstack((self.user_mean_progress[user], u_mean))
            self.user_var_progress[user] = np.vstack((self.user_var_progress[user], u_var))
            self.user_mse_progress[user].append(user_mse)
            
    def save_uncertainty_progress(self, data_name, bandit_name, folder='BanditProgress'):
        fname = data_name + '_' + bandit_name + '_progress.pkl'
        location = os.path.join(folder, fname)
        user_progress = [self.user_mean_progress, self.user_var_progress, self.user_mse_progress]
        _pickle.dump(user_progress, open(location,'wb'))
        
        # reset for next bandit algo
        self.user_mean_progress = {}
        self.user_var_progress = {}
        self.user_mse_progress = {}
        self.current_user = -1
        
        print('saved to ' + location)
        
        
        
        
    
    
