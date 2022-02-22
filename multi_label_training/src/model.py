import torch
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForMultiLabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, multi_task=False):
        super().__init__(config)
        self.multi_task = multi_task
        self.classifier2 = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                targets_per_input_id=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if self.multi_task:

            embeddings = outputs[0]
            # create a vector to retain the output for each token. Shape: [batch_size, seq_len, num_classes]
            logits_per_input_id = torch.zeros((embeddings.shape[0], embeddings.shape[1], self.num_labels)).to(self.device)

            # feed-forward for each token in the sequence and save it in outputs
            for i in range(embeddings.shape[1]):
                # the logits for a single token. Shape: [batch_size, num_classes]
                logit = self.classifier2(self.dropout(embeddings[:, i, :]))

                logits_per_input_id[:, i, :] = logit

            # logits_per_input_id = self.classifier(outputs[0])
            loss_fct_per_input_id = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss_per_input_id = None
            if targets_per_input_id is not None:
                loss_per_input_id = 0
                for instance, target_instance, mask in zip(logits_per_input_id, targets_per_input_id, attention_mask):
                    loss_per_input_id += torch.mean(mask.unsqueeze(1) *
                                                    loss_fct_per_input_id(instance, target_instance))

                loss_per_input_id = loss_per_input_id / logits.shape[0]

            return SequenceClassifierOutput(loss=loss,
                                            logits=logits,
                                            hidden_states=outputs.hidden_states,
                                            attentions=outputs.attentions), (loss_per_input_id, logits_per_input_id)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output



        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


