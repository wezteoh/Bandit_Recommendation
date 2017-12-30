# Bandit Recommendation
In this project we provide a model for Instantaneous Feedback Recommendation
Systems. The observed user ratings from the MovieLens dataset are recorded in
a Ratings Matrix and the missing entries are obtained by factorizing this matrix
using 2 approaches: Probabilistic Matrix Factorization (PMF) and Neural Network
Matrix Factorization (NNMF). Both methods predict the missing entries of the
Ratings matrix as a Gaussian whose mean is a function of dense latent matrices
(with Gaussian priors) representing the item data and user data. The latent’s
posterior distribution parameters are approximated using Stochastic Variational
Inference in both factorizations. We then recast the items in the ratings matrix as
arms in the Multi-armed Bandits context whose predictive reward distributions are
given by that of the dot product of two independent Gaussian vectors. We use these
distributions in conjunction with different Bandit policy functions to provide a
sequence of recommendations to a given user. Finally we measure the quality of our
recommendation sequence using the Regret metric. We found that the exploitation
strategy dominates in the original dataset, while exploratory approaches dominates
in a modified version of the dataset that reflect more ambiguous user preferences.

The final report can be found here:
https://github.com/wezteoh/Bandit_Recommendation/blob/master/Uncertainty%20Guided%20Recommendation%20with%20Bandits.pdf
