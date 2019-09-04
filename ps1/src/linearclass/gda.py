import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    # *** START CODE HERE ***
    x_train1, y_train1 = util.load_dataset('ds1_train.csv', add_intercept=False)
    gda = GDA()
    gda.fit(x_train1, y_train1)
    x_eval1, y_eval1 = util.load_dataset('ds1_valid.csv', add_intercept=False)
    util.plot(x_eval1, y_eval1, gda.theta, 'gda_pred_1.jpg',  correction=1.0)
    predictions = gda.predict(x_eval1)
    np.savetxt('gda_pred_1.txt', predictions)

    #### dataset2
    x_train2, y_train2 = util.load_dataset('ds2_train.csv', add_intercept=False)
    gda = GDA()
    gda.fit(x_train2, y_train2)
    x_eval2, y_eval2 = util.load_dataset('ds2_valid.csv', add_intercept=False)
    util.plot(x_eval2, y_eval2, gda.theta, 'gda_pred_2.jpg',  correction=1.0)
    predictions = gda.predict(x_eval2)
    np.savetxt('gda_pred_2.txt', predictions)

    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        n_examples = x.shape[0]
        features = x.shape[1]
        phi = 0
        mu_1 = 0
        count_mu_1 = 0
        mu_0 = 0
        count_mu_0 = 0
        for i in range(n_examples):
            if y[i] == 1:
                phi = phi + 1/n_examples
                mu_1 += x[i]
                count_mu_1 += 1
                #sigma += ((x[i]- mu_1)*np.transpose(x[i] - mu_1))
            elif y[i] == 0:
                mu_0 += x[i]
                count_mu_0 += 1
                #sigma += ((x[i]- mu_0)*np.transpose(x[i] - mu_0))
        mu_1 = mu_1 / count_mu_1
        mu_0 = mu_0 / count_mu_0
        sigma = np.zeros(shape=(features, features))
        for i in range(n_examples):
            if y[i] == 1:
                sigma += ((x[i]- mu_1).reshape(features,1) @ np.transpose((x[i] - mu_1).reshape(features,1)))
            elif y[i] == 0:
                sigma += ((x[i]- mu_0).reshape(features,1) @ np.transpose((x[i] - mu_0).reshape(features,1)))
        print(" this: ")
        print(x[i].shape)
        print(mu_1.shape)
        print((x[i] - mu_1).shape)
        sigma = (1/n_examples)*sigma
        #sigma = sigma / n_examples

        #print(sigma.shape)
        # Write theta in terms of the parameters
        theta = np.dot(np.linalg.inv(sigma), mu_1) - np.dot(np.linalg.inv(sigma), mu_0)
        print("k", np.dot(np.linalg.inv(sigma), mu_1))
        print("l", np.dot(np.linalg.inv(sigma), mu_0))
        #theta = sigma**(-1)*mu_1 - sigma**(-1)*mu_0
        theta_0 = (1/2) * ((np.transpose(mu_0)) @ np.linalg.inv(sigma) @ mu_0 - (np.transpose(mu_1) @ np.linalg.inv(sigma) @ mu_1)) - np.log((1-phi)/phi)
        self.theta = np.append(theta_0, theta)
        print("theta", self.theta.shape)
        print("mu_0", mu_0.shape)
        print("mu_1", mu_1.shape)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        prediction = sigmoid(x)
        return prediction
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
