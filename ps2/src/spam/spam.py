import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    message = message.lower().split(" ")
    return message
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dictionary = {}
    for sentence in messages:
        for word in set(get_words(sentence)):
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1
    dictionary_new = {}
    i = 0
    for word, count in dictionary.items():
        if count >= 5:
            dictionary_new[word] = i
            i += 1
    return dictionary_new
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    array = np.zeros(shape = (len(messages), len(word_dictionary)))
    for index, sentence in enumerate(messages):
        for word in get_words(sentence):
            if word in word_dictionary:
                array[index, word_dictionary[word]] += 1
    return array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    spam = matrix[labels == 1, :]
    notspam = matrix[labels == 0, :]
    vocab = matrix.shape[1]

    words_1 = np.sum(spam) + vocab
    words_0 = np.sum(notspam) + vocab

    size = np.sum(spam, axis=0) + np.ones(spam.shape[1])
    words_1 = size / words_1
    size = np.sum(notspam, axis=0) + np.ones(notspam.shape[1])
    words_0 = size / words_0

    words_1 = np.log(words_1)
    words_0 = np.log(words_0)
    y_prob_1 = spam.shape[0] / (notspam.shape[0] + spam.shape[0])
    y_prob_0 = notspam.shape[0] / (notspam.shape[0] + spam.shape[0])

    return ((y_prob_0, y_prob_1, words_0, words_1))

    #Some error on alternative code produced lower accuracy on test set.
    '''
    parameter = {}
    matrix_1 = []
    matrix_0 = []
    y_1 = 0
    y_0 = 0
    words = np.sum(matrix)
    vocab_size = matrix.shape[1]
    ys = len(labels)
    spam = matrix[category == 1, :]
    notspam = matrix[category == 0, :]
    spam_lengths = spam.sum(axis = 1)
    nospam_lengths = spam.sum(axis = 1)
    for i, mat in enumerate(matrix):
        if labels[i] == 1:
            matrix_1.append(mat)
            y_1 += 1
    for i, mat in enumerate(matrix):
        if labels[i] == 0:
            matrix_0.append(mat)
            y_0 += 1
    count_1 = np.sum(matrix_1, axis=0)
    count_0 = np.sum(matrix_0, axis=0)
    parameter["phi_1"] = (1 + count_1) / (vocab_size + y_1 * words)
    parameter["phi_0"] = (1 + count_0) / (vocab_size + y_0 * words)
    parameter["phi_y"] = y_1 / ys
    return parameter
    '''
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***

    pred_array = np.zeros(matrix.shape[0])

    ((y_prob_0, y_prob_1, words_0, words_1)) = model

    for num in range(matrix.shape[0]):
        row = matrix[num, :]
        x_0 = np.multiply(words_0, row)
        x_0 = np.exp(np.sum(x_0))
        prob_0 = x_0 * y_prob_0

        x_1 = np.multiply(words_1, row)
        x_1 = np.exp(np.sum(x_1))
        prob_1 = x_1 * y_prob_1
        if prob_1 > prob_0:
            pred_array[num] = 1
        else:
            pred_array[num] = 0
    return pred_array

    # Some error on alternative code produced lower accuracy on test set.
    '''
    predictions = np.zeros(matrix.shape[0])
    log_phi_1 = np.sum( np.log(model["phi_1"])*matrix, axis = 1)
    log_phi_0 = np.sum( np.log(model["phi_0"])*matrix, axis = 1)
    phi = model["phi_y"]
    ratio = np.exp(log_phi_0 + np.log(1 - phi) - log_phi_1 - np.log(phi))
    prob = 1 / (1+ratio)
    predictions[prob > 0.5] = 1
    return predictions
    '''
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Use the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***

    ((y_prob_0, y_prob_1, words_0, words_1)) = model

    temp = words_1 - words_0
    index = temp.argsort()[-5:][::-1].tolist()
    words = []
    for first in index:
        for word in dictionary:
            if dictionary[word] == first:
                words.append(word)
    return words

    # Some error on alternative code produced lower accuracy on test set.
    '''
    spam_words = np.argsort(model["phi_1"] / model["phi_0"])[-5:]
    listed = []
    for key, val in dictionary.items():
        for num in spam_words:
            if val == num:
                listed.append(key)
    return listed
    '''

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best = None
    highest = 0
    for radius in radius_to_consider:
        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(predictions == val_labels)
        if accuracy > highest:
            highest = accuracy
            best = radius
    return best
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)
    print('Size of dictionary: ', len(dictionary))
    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)
    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
