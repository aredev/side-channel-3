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

read_data()