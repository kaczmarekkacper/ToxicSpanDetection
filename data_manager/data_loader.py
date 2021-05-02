import csv
import ast
import itertools
import string


SPECIAL_CHARACTERS = string.whitespace


def read_datafile(filename):
    """Reads csv file with python span list and text."""
    with open(filename, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        data = get_spans(reader)
    return data


def get_spans(reader):
    data = []
    for row in reader:
        fixed = fix_spans(ast.literal_eval(row['spans']), row['text'])
        data.append((fixed, row['text']))
    return data


def contiguous_ranges(span_list):
    output = []
    for _, span in itertools.groupby(
            enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    cleaned = []
    for begin, end in contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned
