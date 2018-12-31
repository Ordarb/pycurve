import pickle


def save_pickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)
