import torch


class Explainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_raw_attn_explanations(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = output.logits
        output_indexes = [i for i, j in enumerate(torch.sigmoid(logits).cpu().detach().numpy()[0]) if j >= .5]

        attentions = output.attentions[0]

        last_attn_layer_weights = attentions[-1]  # take the last attention layer
        avg_last_attn_layer_weights = last_attn_layer_weights.mean(dim=0).unsqueeze(0)  # average through attentions heads
        avg_last_attn_layer_weights[:, 0, 0] = 0  # zeroing CLS token attentions between it and itself
        word_attributions = avg_last_attn_layer_weights[:, 0]  # take CLS toke attentions between it and all others tokens
        return word_attributions, output_indexes
