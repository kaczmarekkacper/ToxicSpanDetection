import random
import statistics
import spacy
import pickle

from loss_functions import f1


class SpacyToxicDetector:
    def __init__(self, train_data, test_data):
        self.spacy_classifier = self.__prepare_spicy()
        self.train_data = train_data
        self.train_data_prepared = self.__prepare_training_data()
        self.test_data = test_data

        self.train_scores = []
        self.test_scores = []
        self.losses = []
        self.iteration = 0

        self.spacy_classifier.begin_training()


    @staticmethod
    def __prepare_spicy():
        print('Configuring SpaCy')
        nlp = spacy.load("en_core_web_sm")
        spicy = spacy.blank('en')
        spicy.vocab.strings.add('TOXIC')
        ner = nlp.create_pipe("ner")  # named entity recognition
        spicy.add_pipe(ner, last=True)
        ner.add_label('TOXIC')

        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        unaffected_pipes = [
            pipe for pipe in spicy.pipe_names
            if pipe not in pipe_exceptions]
        spicy.disable_pipes(*unaffected_pipes)
        return spicy

    def __prepare_training_data(self):
        print('Preparing training data')
        nlp = spacy.load("en_core_web_sm")
        training_data = []
        for n, (spans, text) in enumerate(self.train_data):
            doc = nlp(text)
            ents = self.__spans_to_ents(doc, set(spans), 'TOXIC')
            training_data.append((doc.text, {'entities': ents}))
        return training_data

    def __spans_to_ents(self, doc, spans, label):
        started = False
        left, right, ents = 0, 0, []
        for x in doc:
            if x.pos_ == 'SPACE':
                continue
            if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
                if not started:
                    left, started = x.idx, True
                right = x.idx + len(x.text)
            elif started:
                ents.append((left, right, label))
                started = False
        if started:
            ents.append((left, right, label))
        return ents

    def train(self):
        random.shuffle(self.train_data_prepared)
        losses = {}
        batches = spacy.util.minibatch(
            self.train_data_prepared, size=spacy.util.compounding(
                start=4.0, stop=32.0, compound=1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            self.spacy_classifier.update(texts, annotations, drop=0.5, losses=losses)
        self.losses.append(losses['ner'])
        self.iteration += 1
        return losses['ner']

    def test(self, data):
        scores = []
        for spans, text in data:
            pred_spans = []
            doc = self.spacy_classifier(text)
            for ent in doc.ents:
                pred_spans.extend(range(ent.start_char, ent.start_char + len(ent.text)))
            score = f1.f1(pred_spans, spans)
            scores.append(score)
        mean_score = statistics.mean(scores)
        return mean_score

    def save(self, path='copies/'):
        file = open( path + 'SpaCy' + self.iteration + '.copy', 'w')
        pickle.dump(self, file)

    def get_filename(self):
        return f'SpaCy{self.iteration}.copy'
