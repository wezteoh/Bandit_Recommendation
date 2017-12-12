import numpy as np


def _desparsify(R, MIN_PERC_FILLED_ITEMS, MIN_PERC_FILLED_USERS):
    print("Before desparsify: % of items: ", np.sum(R > 0) / (R.shape[0] * R.shape[1]))

    R_dense = np.copy(R)

    idx = []
    # Remove items
    for j in range(R_dense.shape[1]):
        perc_filled = np.sum(R_dense[:,j] > 0) / R_dense.shape[0]
        if perc_filled >= MIN_PERC_FILLED_ITEMS:
            idx.append(j)
    R_dense = R_dense[:, idx]
    R_dense.shape

    idx = []
    # Remove users
    for i in range(R_dense.shape[0]):
        perc_filled = np.sum(R_dense[i,:] > 0) / R_dense.shape[1]
        if perc_filled >= MIN_PERC_FILLED_USERS:
            idx.append(i)
    R_dense = R_dense[idx, :]
    R_dense.shape

    print("After desparsify: % of items: ", np.sum(R_dense > 0) / (R_dense.shape[0] * R_dense.shape[1]))
    print("Size: ", R_dense.shape)
    return R_dense
    
def remove_polarized_ratings(R):
    POLARITY_THRESHOLD = 0.8 # remove all movies where at least this many people liked or disliked the movie.

    keep_js = []
    for j in range(R.shape[1]):
        negatives = np.sum((R[:,j] == 1) | (R[:,j] == 2))
        undecided = np.sum(R[:,j] == 3)
        positives = np.sum((R[:,j] == 4) | (R[:,j] == 5))

        if positives+negatives > 0 and \
            (negatives/(positives+negatives) > POLARITY_THRESHOLD or
            positives/(positives+negatives) > POLARITY_THRESHOLD):
                continue

        keep_js.append(j)

    R_ = np.copy(R)
    R_ = R_[:, keep_js]
    return R_
    
def preprocess_data(R, MIN_PERC_FILLED_ITEMS=0.1, MIN_PERC_FILLED_USERS=0.25):
    return _desparsify(remove_polarized_ratings(R), MIN_PERC_FILLED_ITEMS, MIN_PERC_FILLED_USERS)
    
def prepare_test_users(R, upper_threshold = 800, lower_threshold=65, test_size=20, seed=1337):
    np.random.seed(seed)
    rating_density = np.sum((R>0), axis=1)    
    qualified = np.bitwise_and(rating_density>lower_threshold, rating_density<=upper_threshold)
    dense_users = np.where(qualified)[0]
    
    return np.random.choice(dense_users, test_size)
    
    
    
    
