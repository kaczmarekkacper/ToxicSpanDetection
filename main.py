# https://stackoverflow.com/questions/66149878/e053-could-not-read-config-cfg-resumeparser
import random
import statistics

import configparser

from data_manager import data_loader
from painter import painter
from spicy_toxic_detector import spicy_toxic_detector
from serializer import serializer


def main():
    print('Loading configurations')
    config = configparser.ConfigParser()
    config.read('configuration.config')
    epoch = int(config['DEFAULT']['epoch'])
    data_path = config['DEFAULT']['data_path']
    load_save = config['DEFAULT']['load_save'] == "True"
    load_filename = config['DEFAULT']['load_filename']
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
        filename = spacy_classifier.get_filename()
        serializer.save(spacy_classifier, filename)

        train_score = spacy_classifier.test(spacy_classifier.train_data)
        spacy_classifier.train_scores.append(train_score)

        test_score = spacy_classifier.test(spacy_classifier.test_data)
        spacy_classifier.test_scores.append(test_score)

        print(f'Iteration nr {iteration}, train_score: {train_score}, test_score: {test_score}, losses : {loss}')
    painter.draw_one_data_plot(spacy_classifier.losses, 'Loss funtion value', 'epoch', 'Loss value')
    painter.draw_two_data_plot(spacy_classifier.train_scores, spacy_classifier.test_scores,
                               'Comparison of train and test scores', 'epoch', 'score', ['train_data', 'test_data'])


if __name__ == '__main__':
    main()
