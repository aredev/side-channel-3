import scipy.io
import numpy
import scipy.stats

__author__ = "Tom Sandmann (s4330048) & Abdullah Rasool (s4350693)"

# Present sbox
sbox = dict([
    (0, 12),
    (1, 5),
    (2, 6),
    (3, 11),
    (4, 9),
    (5, 0),
    (6, 10),
    (7, 13),
    (8, 3),
    (9, 14),
    (10, 15),
    (11, 8),
    (12, 4),
    (13, 7),
    (14, 1),
    (15, 2),
])


def read_data():
    messages = scipy.io.loadmat('data.mat')['messages']
    traces = scipy.io.loadmat('data.mat')['traces']

    return messages, traces


def apply_selection_function(messages, traces):
    keys = range(0, 4)

    values_zero = numpy.zeros([len(messages), 4])
    idx_zero = [0, 0, 0, 0]
    values_one = numpy.zeros([len(messages), 4])
    idx_one = [0, 0, 0, 0]

    skewness_zero = [0, 0, 0, 0]
    skewness_one = [0, 0, 0, 0]

    for i, m in enumerate(messages):
        for k in keys:
            mbin = format(m[0], "04b")
            kbin = format(k, "04b")

            selection_value = (int(mbin[1]) + int(kbin[1])) * (int(mbin[2]) + int(kbin[2])) + int(mbin[2]) + int(
                mbin[3])
            selection_value = selection_value % 2

            if selection_value == 0:
                values_zero[idx_zero[k]][k] = traces[i][0]
                idx_zero[k] += 1
            else:
                values_one[idx_one[k]][k] = traces[i][0]
                idx_one[k] += 1

            skewness_zero[k] = scipy.stats.skew(values_zero[:, k])
            skewness_one[k] = scipy.stats.skew(values_one[:, k])

    print(skewness_one)
    print(skewness_zero)

    return values_zero, values_one


def apply_second_selection_function(messages, traces):
    keys = range(0, 4)

    values_zero = numpy.zeros([len(messages), 4])
    idx_zero = [0, 0, 0, 0]
    values_one = numpy.zeros([len(messages), 4])
    idx_one = [0, 0, 0, 0]

    skewness_zero = [0, 0, 0, 0]
    skewness_one = [0, 0, 0, 0]

    for i, m in enumerate(messages):
        for k in keys:
            mbin = format(m[0], "04b")
            kbin = format(k, "04b")

            selection_value = int(mbin[0] + kbin[0]) * int((mbin[2] + kbin[2])) * int((mbin[1] + kbin[1]))
            selection_value += int((mbin[0] + kbin[0])) * int((mbin[3] + kbin[3])) * int((mbin[1] + kbin[1]))
            selection_value += int((mbin[3] + kbin[3])) * int((mbin[1] + kbin[1]))
            selection_value += int((mbin[1] + kbin[1]))
            selection_value += int((mbin[0] + kbin[0])) * int((mbin[2] + kbin[2])) * int((mbin[3] + kbin[3]))
            selection_value += int((mbin[2] + kbin[2])) * int((mbin[3] + kbin[3]))
            selection_value += int((mbin[3] + kbin[3]))
            selection_value = selection_value % 2

            if selection_value == 0:
                values_zero[idx_zero[k]][k] = traces[i][0]
                idx_zero[k] += 1
            else:
                values_one[idx_one[k]][k] = traces[i][0]
                idx_one[k] += 1

            skewness_zero[k] = scipy.stats.skew(values_zero[:, k])
            skewness_one[k] = scipy.stats.skew(values_one[:, k])

    print(skewness_one)
    print(skewness_zero)

    return values_zero, values_one


def selection_function(messages, traces):
    """
    Apply the DoM attack but with the skewness instead
    :param messages: 
    :return: 
    """

    zero_values, one_values = apply_selection_function(messages, traces)
    results = []

    for i in range(0, 4):
        key_i_zero_traces = zero_values[:, i] #Get column of key with value i
        key_i_zero_traces = numpy.asarray(list(filter(lambda a: a != 0, key_i_zero_traces)))

        key_i_one_traces = one_values[:, i]
        key_i_one_traces = numpy.asarray(list(filter(lambda a: a != 0, key_i_one_traces)))

        skewness_zero = scipy.stats.skew(key_i_zero_traces)
        skewness_one = scipy.stats.skew(key_i_one_traces)
        skewness_delta = abs(skewness_one - skewness_zero)

        results.append((skewness_delta, i))

    results = sorted(results, key=lambda tup: tup[1])
    print(results)

    zero_values, one_values = apply_second_selection_function(messages, traces)
    results = []

    for i in range(0, 4):
        key_i_zero_traces = zero_values[:, i]  # Get column of key with value i
        print(key_i_zero_traces)

        key_i_one_traces = one_values[:, i]

        skewness_zero = scipy.stats.skew(key_i_zero_traces)
        skewness_one = scipy.stats.skew(key_i_one_traces)
        skewness_delta = abs(skewness_one - skewness_zero)

        results.append((skewness_delta, i))

    results = sorted(results, key=lambda tup: tup[1])
    print(results)





messages, traces = read_data()
selection_function(messages, traces)
