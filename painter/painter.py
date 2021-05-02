import matplotlib.pyplot as plt


def draw_one_data_plot(data, title, xlabel, ylabel):
    x = range(1, len(data) + 1)
    plt.plot(x, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def draw_two_data_plot(train, test, title, xlabel, ylabel, legend):
    x1 = range(1, len(train) + 1)
    x2 = range(1, len(test) + 1)
    plt.plot(x1, train, 'g', x2, test, 'b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.show()
