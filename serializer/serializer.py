import pickle


def save(obj, filename, path='objects/'):
    file = open(path + filename, 'wb+')
    pickle.dump(obj, file)


def load(filename, path='objects/'):
    file = open(path + filename, 'rb')
    obj = pickle.load(file)
    return obj
