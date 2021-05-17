# https://stackoverflow.com/questions/66149878/e053-could-not-read-config-cfg-resumeparser
import configparser
import numpy as np

from data_manager import data_loader
from painter import painter
from models import spicy_toxic_detector
from models import dict_toxic_detector
from serializer import serializer

def spicy_main(config):
    epoch = int(config['SPICY']['epoch'])
    data_path = config['SPICY']['data_path']
    load_save = config['SPICY']['load_save'] == "True"
    load_filename = config['SPICY']['load_filename']
    if load_save:
        spacy_classifier = serializer.load(load_filename)
    else:
        print('Loading training data')
        train = data_loader.read_datafile(data_path + 'train.csv')

        print('Loading test data')
        test = data_loader.read_datafile(data_path + 'test.csv')

        spacy_classifier = spicy_toxic_detector.SpacyToxicDetector(train, test)

    print('Training')
    for iteration in range(spacy_classifier.iteration + 1, spacy_classifier.iteration + epoch + 1):
        loss = spacy_classifier.train()

        train_score = spacy_classifier.test(spacy_classifier.train_data)
        spacy_classifier.train_scores.append(train_score)

        test_score = spacy_classifier.test(spacy_classifier.test_data)
        spacy_classifier.test_scores.append(test_score)

        filename = spacy_classifier.get_filename()
        serializer.save(spacy_classifier, filename)

        print(f'Iteration nr {iteration}, train_score: {train_score}, test_score: {test_score}, losses : {loss}')
    losses_running_mean = running_mean(spacy_classifier.losses, 10)
    train_score_running_mean = running_mean(spacy_classifier.train_scores, 10)
    test_score_running_mean = running_mean(spacy_classifier.test_scores, 10)
    painter.draw_one_data_plot(losses_running_mean, 'Loss function value', 'epoch', 'Loss value')
    painter.draw_two_data_plot(train_score_running_mean, test_score_running_mean,
                               'Comparison of train and test scores', 'epoch', 'score', ['train_data', 'test_data'])

def dict_main(config):
    data_path = config['DICT']['data_path']
    print('Loading training data')
    train = data_loader.read_datafile(data_path + 'train.csv')

    print('Loading test data')
    test = data_loader.read_datafile(data_path + 'test.csv')


    dict_model = dict_toxic_detector.DictToxicDetector(train, test)

    print('Training')
    dict_model.train()

    print('Testing')
    print(dict_model.test())



def main():
    print('Loading configurations')
    config = configparser.ConfigParser()
    config.read('configuration.config')
    model = config['DEFAULT']['model']
    if model == 'SPICY':
        spicy_main(config)
    elif model == 'DICT':
        dict_main(config)


def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


if __name__ == '__main__':
    main()
