from loss_functions import f1
import statistics

class DictToxicDetector:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.dictionary = set()


    def train(self):
        for element in self.train_data:
            spans = self.get_spans(element[0])
            sentence = element[1]
            for span in spans:
                start_idx = span[0]
                end_idx = span[-1] + 1
                toxic_sentence = sentence[start_idx:end_idx]
                self.dictionary.add(toxic_sentence.lower())

    def get_spans(self, span_list):
        spans = []
        if not span_list:
            return spans
        start_idx = span_list[0]
        prev_idx = span_list[0]
        for i in range(1, len(span_list)):
            current_idx = span_list[i]
            if current_idx - 1 == prev_idx:
                prev_idx = current_idx
            else:
                spans.append((start_idx, prev_idx))
                start_idx = current_idx
                prev_idx = current_idx
        if start_idx != prev_idx:
            spans.append((start_idx, prev_idx))
        return spans

    def test(self):
        scores = []
        for spans, sentence in self.test_data:
            pred_spans = []
            lower_sentence = sentence.lower()
            for toxic_sentance in self.dictionary:
                start_idx = lower_sentence.find(toxic_sentance)
                if start_idx >= 0:
                    end_idx = start_idx + len(toxic_sentance) - 1
                    pred_spans.extend(range(start_idx, end_idx))
            score = f1.f1(pred_spans, spans)
            scores.append(score)
        mean_score = statistics.mean(scores)
        return mean_score



