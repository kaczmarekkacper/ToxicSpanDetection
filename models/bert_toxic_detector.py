import string

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from keras.preprocessing.sequence import pad_sequences


class BertToxicDetector:
    def __init__(self, train_data, test_data, epochs, max_grad_norm, batch_size):
        self.tags = {'O': 0, 'B-TOXIC': 1, 'I-TOXIC': 2}
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.train_data = self.__prepare_data(train_data)
        self.test_data = self.__prepare_data(test_data)
        self.optimizer = None
        self.scheduler = None

        self.bert_classifier = self.__prepare_bert()

        self.train_scores = []
        self.test_scores = []
        self.losses = []
        self.iteration = 0

    def __prepare_bert(self):
        print('Configuring Bert')
        model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=len(self.tags),
            output_attentions=False,
            output_hidden_states=False
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_data) * self.epochs

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        return model

    def __prepare_data(self, data):
        print('Preparing data')
        preprocessed_data = []
        sentences = []
        labels = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        for spans, sentence in data:
            tokenized_sentence = tokenizer.tokenize(sentence)
            sentences.append(tokenized_sentence)
            idx = 0
            started = False
            labs = []
            for word in tokenized_sentence:
                if idx in spans:
                    if started:
                        labs.append('I-TOXIC')
                    else:
                        labs.append('B-TOXIC')
                        started = True
                else:
                    labs.append('O')
                    started = False
                idx += len(word.replace("#", ""))
                idx += 0 if word.find("#") != -1 or string.punctuation else 1
            labels.append(labs)
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in sentences],
                                  dtype="long", value=0.0,
                                  truncating="post", padding="post")

        tags = pad_sequences([[self.tags[l] for l in lab] for lab in labels],
                             padding="post",
                             dtype="long", truncating="post")

        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

        inputs_tensor = torch.tensor(input_ids).type(torch.LongTensor)
        tags_tensor = torch.tensor(tags).type(torch.LongTensor)
        masks_tensor = torch.tensor(attention_masks).type(torch.LongTensor)

        dataset = TensorDataset(inputs_tensor, masks_tensor, tags_tensor)
        sampler = RandomSampler(dataset)
        preprocessed_data = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return preprocessed_data

    def train(self):
        self.bert_classifier.train()
        total_loss = 0
        for step, batch in enumerate(self.train_data):
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            self.bert_classifier.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = self.bert_classifier(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=self.bert_classifier.parameters(), max_norm=self.max_grad_norm)
            # update parameters
            self.optimizer.step()
            # Update the learning rate.
            self.scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(self.train_data)

        # Store the loss value for plotting the learning curve.
        self.losses.append(avg_train_loss)
        return avg_train_loss

    def test(self, data):
        self.bert_classifier.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in data:
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = self.bert_classifier(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(data)
        # validation_loss_values.append(eval_loss)
        # print("Validation loss: {}".format(eval_loss))
        #TODO poprawić
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


    def save(self, path='copies/'):
        file = open(path + 'Bert' + self.iteration + '.copy', 'w')
        pickle.dump(self, file)

    def get_filename(self):
        return f'SpaCy{self.iteration}-dropout-{self.dropout}.copy'
