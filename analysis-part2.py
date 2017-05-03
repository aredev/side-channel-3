import scipy.io
import numpy
import scipy.stats
import math
import matplotlib.mlab as mlab
from scipy.stats import norm


__author__ = "Tom Sandmann (s4330048) & Abdullah Rasool (s4350693)"


def read_data():
    file = scipy.io.loadmat('dataset.mat')
    train0 = file['train_set0']
    train1 = file['train_set1']
    test0 = file['test_set0']
    test1 = file['test_set1']

    return train0, train1, test0, test1


def compute_mean_vector(vector):
    """
    Compute the mean vector of a vector of vector [[...]] => [m1, m2, ..., mn]
    :param vector: 
    :return: 
    """

    mean_vector = []
    for v in range(len(vector[0])):
        mean_vector.append(numpy.mean(vector[:, v]))

    return mean_vector


def build_reduced_template(train0, train1):
    """
    Compute the mean for each training set, that is the training set for 0 or one
    :param train0: set of training data for bit = 0
    :param train1: set of training data for bit = 1
    :return: the templates t0 and t1
    """

    t0 = compute_mean_vector(train0)
    t1 = compute_mean_vector(train1)
    return t0, t1


def vector_substraction(v1, v2):
    return [a_i - b_i for a_i, b_i in zip(v1, v2)]


def vector_addition(v1, v2):
    return [a_i + b_i for a_i, b_i in zip(v1, v2)]


def reduced_template_matching(t0, test0, t1, test1):
    incorrect0 = 0
    incorrect1 = 0

    for i, v in enumerate(test0):
        vsubt0 = vector_substraction(v, t0)
        vsubt1 = vector_substraction(v, t1)
        score0 = numpy.asmatrix(vsubt0).dot(numpy.asmatrix(vsubt0).transpose())
        score1 = numpy.asmatrix(vsubt1).dot(numpy.asmatrix(vsubt1).transpose())
        if score0 <= score1:
            # It is probably bit 1, which is incorrect for this test set
           incorrect0 += 1

    for i, v in enumerate(test1):
        vsubt0 = vector_substraction(v, t0)
        vsubt1 = vector_substraction(v, t1)
        score0 = numpy.asmatrix(vsubt0).dot(numpy.asmatrix(vsubt0).transpose())
        score1 = numpy.asmatrix(vsubt1).dot(numpy.asmatrix(vsubt1).transpose())
        if score0 > score1:
            # It is probably bit 0, which is incorrect for this test set
            incorrect1 += 1

    misrate0 = incorrect0/len(test0)
    misrate1 = incorrect1/len(test1)

    print("Reduced template: misclassification rate for O_0: ", misrate0)
    print("Reduced template: misclassification rate for O_1: ", misrate1)


def build_univariate_template(train0, train1):
    """
    Build the univariate template
    :param train0: 
    :param train1: 
    :return: 
    """
    # First lets find the PoI, by calculating the absolute difference of means
    # 1. Calculate the means of the training sets
    mean_0 = compute_mean_vector(train0)
    mean_1 = compute_mean_vector(train1)

    # 2. Calculate the mean of each sample training sample
    submeans = vector_substraction(mean_0, mean_1) # Calculate the difference of all means
    sorted_submeans = sorted(((value, index) for index, value in enumerate(submeans)), reverse=True) # Take the higest difference
    poiIndex = sorted_submeans[0][1]

    # 3. The one which has the largest difference to the first mean is the PoI.
    mu0, sigma0 = compute_mle_params(train0[:, poiIndex])
    mu1, sigma1 = compute_mle_params(train1[:, poiIndex])

    print(mu0, sigma0, mu1, sigma1)

    return mu0, sigma0, mu1, sigma1


def compute_mle_params(poi):
    mu = numpy.sum(poi)/len(poi)
    sigma_sq = sum((poi - mu)**2)*1/(len(poi)-1)

    sigma = math.sqrt(sigma_sq)
    return mu, sigma


def univariate_test_templates(mu0, sigma0, mu1, sigma1, test0, test1):
    incorrect0 = 0
    incorrect1 = 0

    for t in test0:
        incorrect0 += univariate_template_matching(t, mu0, sigma0, mu1, sigma1, True)

    for t in test1:
        incorrect1 += univariate_template_matching(t, mu0, sigma0, mu1, sigma1, False)

    misrate0 = incorrect0/(len(test0)*1301)
    misrate1 = incorrect1/(len(test1)*1301)

    print("Univariate template: misclassification rate for O_0: ", misrate0)
    print("Univariate template: misclassification rate for O_1: ", misrate1)


def univariate_template_matching(rtest, mu0, sigma0, mu1, sigma1, bigger):
    """
    We get the test value and match whether it belongs to zero or one, given mu and sigma.
    :param rtest: Test value 
    :param mu: 
    :param sigma: 
    :return: true when the test value belongs to 1 and false if it belongs to 0.
    """

    ratios = norm.pdf(rtest, mu0, sigma0) / norm.pdf(rtest, mu1, sigma1) > 1

    if bigger:
        # The ratio should be bigger than 1
        return list(ratios).count(True)
    else:
        return list(ratios).count(False)


def build_multivariate_template(train0, train1):
    mean_o0 = compute_mean_vector(train0)
    mean_o1 = compute_mean_vector(train1)
    tbar = vector_addition(mean_o0, mean_o1)
    tbar = [x * 0.5 for x in tbar]
    mean_o0subtbar = [a_i - b_i for a_i, b_i in zip(mean_o0, tbar)]
    mean_o1subtbar = [a_i - b_i for a_i, b_i in zip(mean_o1, tbar)]

    mean_o0subtbar = numpy.asmatrix(mean_o0subtbar).dot(numpy.asmatrix(mean_o0subtbar).transpose())
    print(mean_o0subtbar)
    mean_o1subtbar = mean_o1subtbar * numpy.transpose(mean_o1subtbar)

    twob = mean_o0subtbar+mean_o1subtbar
    print(len(twob))
    # print(twob)
    #
    #
    # print(b)
    # svd = numpy.linalg.svd(b)
    # print(svd)


train0, train1, test0, test1 = read_data()
# t0, t1 = build_reduced_template(train0, train1)
# reduced_template_matching(t0, test0, t1, test1)
mu0, sigma0, mu1, sigma1 = build_univariate_template(train0, train1)
univariate_test_templates(mu0, sigma0, mu1, sigma1, test0, test1)
# build_multivariate_template(train0, train1)
