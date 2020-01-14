import streamlit as st
import pandas as pd

st.title('DTrees - Classifier')
st.write('**Data**')

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]
header = ['color', 'diameter', 'label']

df = pd.DataFrame(training_data, columns=header)
st.write(df)

def unique_values(rows, col):
    return set([row[col] for row in rows])

# print(unique_values(training_data, 0))

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label]+=1
    return counts

# print(class_counts(training_data))

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

# print(is_numeric(9))

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition=">="
        return f'Is {header[self.column]} {condition} {str(self.value)} ?'


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -=prob_of_lbl**2
    return impurity

no_mixing = [
    ['Apple'],
    ['Apple']
]
print(gini(no_mixing))


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain>= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

best_gain, best_question = find_best_split(training_data)
print(f'Here is the best gain {best_gain} with this question {best_question}')
