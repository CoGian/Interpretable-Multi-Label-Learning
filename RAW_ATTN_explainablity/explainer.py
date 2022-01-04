import torch
import numpy as np


class Explainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_raw_attn_explanations(self, input_ids, attention_mask, starting_layer=0, ending_layer=6):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = output.logits
        output_indexes = [i for i, j in enumerate(torch.sigmoid(logits).cpu().detach().numpy()[0]) if j >= .5]

        attn_weights = np.array([item.cpu().detach().numpy() for item in output.attentions[starting_layer: ending_layer]])
        word_attributions = attn_weights.mean(axis=0).mean(axis=1).mean(axis=1)
        # word_attributions = attn_weights.mean(axis=0).mean(axis=1)[:, 0] # taking only CLS tokeν attentions between it and all others tokens
        word_attributions[:, -1] = 0
        word_attributions[:, 0] = 0

        return word_attributions, output_indexes, output
