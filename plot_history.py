from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import math


def parse_history(filename):
    history_data = {'epoch': [], 'acc': []}
    with open(filename, 'r') as history_file:
        lines = history_file.readlines()
    lines = lines[1:]

    def parse_line(line):
        parts = line.split(',')
        epoch = int(parts[1])
        acc = float(parts[2])
        history_data['epoch'].append(epoch)
        history_data['acc'].append(acc)

    for line in lines:
        parse_line(line)
    return history_data


def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n, )) / n, mode=mode))


def round_down_by(data, factor):
    data = int(data * 1000)
    factor = int(factor * 1000)
    return float(math.floor(data / factor)) * factor / 1000.


def round_up_by(data, factor):
    data = int(data * 1000)
    factor = int(factor * 1000)
    return float(math.ceil(data / factor)) * factor / 1000.


def plot_history(history_data):
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('validation_acc', fontsize=14)
    plt.plot(history_data['epoch'],
             history_data['acc'],
             color="deeppink",
             linewidth=1,
             linestyle=':',
             label='raw accuray',
             marker='o')
    acc_moving_avg = np_move_avg(history_data['acc'], 5)
    plt.plot(history_data['epoch'][0:-2],
             acc_moving_avg[0:-2],
             color="darkblue",
             linewidth=1,
             linestyle='--',
             label='moving average accuray',
             marker='+')
    acc_min, acc_max = np.amin(acc_moving_avg), np.amax(acc_moving_avg)
    factor = 0.05
    acc_min = round_down_by(acc_min, factor)
    acc_max = round_up_by(acc_max, factor)
    plt.yticks(np.arange(acc_min, acc_max + factor, factor))
    plt.grid()
    plt.legend()
    plt.show()


history_data = parse_history('cifar10_resnet_validation.csv')
plot_history(history_data)
#pprint(history_data)