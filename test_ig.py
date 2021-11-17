from transformers import AutoModelForSequenceClassification, AutoTokenizer
from IG_BERT_explainability.multi_label_sequence_classification import MultiLabelSequenceClassificationExplainer
import json
import re
from sklearn import preprocessing
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12")
    model = AutoModelForSequenceClassification.from_pretrained("HoC_models/HoC_ncbi_bert_pubmed/")
    model.to(device)
    model.eval()

    multilabel_explainer = MultiLabelSequenceClassificationExplainer(model=model, tokenizer=tokenizer)

    topics = []
    with open("Datasets/HoC/topics.json", "r") as f:
        for label in f.readlines():
            topics.append(label.strip())
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([topics])

    with open("Datasets/HoC/val.json", "r") as fval:
        val_dataset = json.load(fval)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    scores_per_label = {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in topics}
    for item in tqdm(val_dataset):

        text = item["text"].lower()
        text = re.sub(r'(\S)\.', r'\g<1>,', text)
        text = re.sub(r'\.(\S)', r',\g<1>', text)

        gold_labels = mlb.transform([item["labels"]])[0]
        gold_indexes = [i for i, j in enumerate(gold_labels) if j >= 1]

        word_attributions = multilabel_explainer(text=text, n_steps=8, internal_batch_size=4)

        output_indexes = multilabel_explainer.selected_indexes

        try:
            for i, output_index in enumerate(output_indexes):
                if output_index not in gold_indexes:
                    continue
                sentences_expl = []
                sent_expl = []
                for word in enumerate(word_attributions[i]):
                    sent_expl.append(word[1][1])
                    if word[1][0] == ".":
                        sentences_expl.append(sent_expl)
                        sent_expl = []

                sent_scores = []
                for sent_expl in sentences_expl:
                    sent_scores.append(np.mean(sent_expl))

                sent_scores = np.array(sent_scores)
                sent_scores = (sent_scores - sent_scores.min()) / (sent_scores.max() - sent_scores.min())

                one_hot = np.zeros((1, len(gold_labels)), dtype=np.float32)
                one_hot[0, output_index] = 1

                gold_label = mlb.inverse_transform(one_hot)[0][0]

                for index, score in enumerate(sent_scores):
                    if score > 0.9:
                        if gold_label in item["labels_per_sentence"][index]:
                            tp += 1
                            scores_per_label[gold_label]["tp"] += 1
                        else:
                            fp += 1
                            scores_per_label[gold_label]["fp"] += 1
                    else:
                        if gold_label in item["labels_per_sentence"][index]:
                            fn += 1
                            scores_per_label[gold_label]["fn"] += 1
                        else:
                            tn += 1
                            scores_per_label[gold_label]["tn"] += 1
        except IndexError:
            print("4444 error for item: ", item["pmid"])

    print("tp", tp)
    print("fp", fp)
    print("tn", tn)
    print("fn", fn)
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", (2*recall*precision)/(recall+precision)) #micro

    metrics_per_labels = {}
    for label in topics:
        recall = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fn"])
        precision = scores_per_label[label]["tp"] / (scores_per_label[label]["tp"] + scores_per_label[label]["fp"])
        f1 = (2*recall*precision)/(recall+precision)
        metrics_per_labels[label] = {"recall": recall, "precision": precision, "f1": f1}

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(metrics_per_labels))