import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    '''
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept = True)
    clf = LogisticRegression()
    predicted_theta = clf.fit(x_train, y_train)
    np.savetxt(save_path, clf.predict(x_eval, predicted_theta), delimiter = ',')

    util.plot(x_eval, y_eval, predicted_theta, "graph.png")
    '''
    #DATASET 1
    x_train, y_train = util.load_dataset('ds1_train.csv', add_intercept=True)
    x_eval, y_eval = util.load_dataset('ds1_valid.csv', add_intercept=True)
    clf = LogisticRegression()
    parameters = clf.fit(x_train, y_train)
    util.plot(x_eval, y_eval, parameters, 'logreg_pred_1.jpg')
    predictions = clf.predict(x_eval, parameters)
    np.savetxt('logreg_pred_1.txt', predictions, delimiter = ',')

    #DATASET 1
    x_train, y_train = util.load_dataset('ds2_train.csv', add_intercept=True)
    x_eval, y_eval = util.load_dataset('ds2_valid.csv', add_intercept=True)
    clf = LogisticRegression()
    parameters = clf.fit(x_train, y_train)
    util.plot(x_eval, y_eval, parameters, 'logreg_pred_2.jpg')
    predictions = clf.predict(x_eval, parameters)
    np.savetxt('logreg_pred_2.txt', predictions, delimiter = ',')

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
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

    def fit(self, x_train, y_train):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        #Sigmoid function
        def sigmoid(x):
            return 1 / (1+np.exp(-x) )
        #Gradient function
        def new(theta, x):
            return np.dot(x, theta)

        def gradient(theta, x, y):
            m = x.shape[0]
            return (1/m) * np.dot(x.T, sigmoid(new(theta, x)) - y)

        def hessian(theta, x, y):
            hessian = np.zeros((x.shape[1], x.shape[1]))
            z = y * x.dot(theta)
            for i in range(hessian.shape[0]):
                for j in range(hessian.shape[0]):
                    if i <= j:
                        hessian[i][j] = np.mean(sigmoid(z) * (1 - sigmoid(z)) * x[:, i] * x[:, j])
                        if i != j:
                            hessian[j][i] = hessian[i][j]
            return hessian

        def newton(theta0, x, y, grad, hess, eps):
            theta = theta0
            difference = 1
            max = 0
            while difference > eps and max < self.max_iter:
                theta_prev = theta.copy()
                theta -= self.step_size * np.linalg.inv(hess(theta, x, y)).dot(grad(theta, x, y))
                difference = np.linalg.norm(theta - theta_prev, ord = 1)
                max += 1
            return theta

        #theta_final = newton(thetazeros, x, y, gradient, hessian)
        #return theta_final
        theta_0 = np.zeros(x_train.shape[1])
        theta = newton(theta_0, x_train, y_train, gradient, hessian, self.eps)
        return theta
        # *** END CODE HERE ***

    def predict(self, x, parameters):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        return sigmoid(np.dot(x, parameters))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
