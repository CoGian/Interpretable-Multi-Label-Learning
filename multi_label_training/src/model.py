import torch
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForMultiLabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, multi_task=False):
        super().__init__(config)
        self.multi_task = multi_task

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
            logits_per_input_id = self.classifier(outputs[0])
            loss_fct_per_input_id = torch.nn.BCEWithLogitsLoss()
            loss_per_input_id = loss_fct_per_input_id(logits_per_input_id.view(-1, self.config.max_position_embeddings, self.num_labels),
                            targets_per_input_id.float().view(-1, self.config.max_position_embeddings, self.num_labels))

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


