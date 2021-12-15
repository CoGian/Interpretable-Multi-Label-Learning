import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from IG_BERT_explainability.multi_label_sequence_classification import MultiLabelSequenceClassificationExplainer
import json
import re
from sklearn import preprocessing
import torch
from tqdm import tqdm
from utils.metrics import update_sentence_metrics, print_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', '-t', help='The percentile threshold of accepting a sentence as rationale')
    args = parser.parse_args()
    threshold = int(args.threshold)

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

    scores = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    scores_per_label = {label: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for label in topics}

    for item in tqdm(val_dataset):

        text = item["text"].lower()
        text = re.sub(r'(\S)\.', r'\g<1>,', text)
        text = re.sub(r'\.(\S)', r',\g<1>', text)

        gold_labels = mlb.transform([item["labels"]])[0]
        gold_indexes = [i for i, j in enumerate(gold_labels) if j >= 1]

        word_attributions_per_pred_class = multilabel_explainer(text=text, n_steps=16, internal_batch_size=4)

        output_indexes = multilabel_explainer.selected_indexes

        try:
            for i, output_index in enumerate(output_indexes):
                if output_index not in gold_indexes:
                    continue
                sentences_expl = []
                sent_expl = []
                for word in enumerate(word_attributions_per_pred_class[i]):
                    sent_expl.append(word[1][1])
                    if word[1][0] == ".":
                        sentences_expl.append(sent_expl)
                        sent_expl = []

                update_sentence_metrics(
                    sentences_expl,
                    gold_labels,
                    output_index,
                    scores,
                    scores_per_label,
                    mlb,
                    item,
                    threshold
                )
        except IndexError:
            print("4444 error for item: ", item["pmid"])

    print_metrics(scores, scores_per_label, topics)
