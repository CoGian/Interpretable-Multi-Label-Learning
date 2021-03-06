import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from IG_BERT_explainability.explainer import BaseExplainer
from IG_BERT_explainability.attributions import LIGAttributions
from IG_BERT_explainability.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)
from utils.metrics import max_abs_scaling, min_max_scaling

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class MultiLabelSequenceClassificationExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type
    `{MODEL_NAME}ForSequenceClassification` from the Transformers package.
    Calculates attribution for `text` using the given model
    and tokenizer.
    This explainer also allows for attributions with respect to a particlar embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
        weight_aggregation="mean_pos"
    ):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".
            custom_labels (List[str], optional): Applies custom labels to label2id and id2label configs.
                                                 Labels must be same length as the base model configs' labels.
                                                 Labels and ids are applied index-wise. Defaults to None.
        Raises:
            AttributionTypeNotSupportedError:
        """
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        self.attributions_list: List[LIGAttributions] = []
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

        self.internal_batch_size = None
        self.n_steps = 50

        self.weight_aggregation = weight_aggregation

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    def predicted_labelset_indexes(self):
        "Returns predicted labelset indexes for model with last calculated `input_ids`"
        if len(self.input_ids) > 0:
            # we call this before _forward() so it has to be calculated twice
            output = self.model(self.input_ids)
            self.output = output
            preds = output[0]
            output_indexes = [i for i, j in enumerate(torch.sigmoid(preds).cpu().detach().numpy()[0]) if j >= .5]
            return output_indexes
        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    @property
    def word_attributions(self) -> list:
        '''
        Returns the scaled (0,1) word attributions for model and the text provided. Raises error if attributions not calculated.
        :return: word_attributions
        '''

        word_attributions_list = []
        for attributions in self.attributions_list:
            if attributions is not None:
                word_attributions = []
                words = []
                for word, value in attributions.word_attributions:

                    if value < 0:
                        if self.weight_aggregation == "mean_pos":
                            word_attributions.append(0)
                        elif self.weight_aggregation == "mean_abs":
                            word_attributions.append(abs(value))
                        else:
                            word_attributions.append(value)
                    else:
                        word_attributions.append(value)
                    words.append(word)
                word_attributions = np.array(word_attributions)

                if self.weight_aggregation == "mean":
                    scaled_word_attributions = max_abs_scaling(word_attributions)
                else:
                    scaled_word_attributions = min_max_scaling(0, 1, word_attributions)

                word_attributions = [(word, value) for word, value in zip(words, scaled_word_attributions)]

                word_attributions_list.append(word_attributions)
            else:
                raise ValueError(
                    "Attributions have not yet been calculated. Please call the explainer on text first."
                )
        return word_attributions_list

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        if self.accepts_position_ids:
            preds = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            preds = preds[0]

        else:
            preds = self.model(input_ids, attention_mask)[0]

        return torch.sigmoid(preds)[:, self.selected_index]

    def _calculate_attributions(  # type: ignore
        self, embeddings: Embedding):
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        self.selected_indexes = self.predicted_labelset_indexes()

        for index in self.selected_indexes:
            self.selected_index = index
            reference_tokens = [
                token.replace("??", "") for token in self.decode(self.input_ids)
            ]
            lig = LIGAttributions(
                self._forward,
                embeddings,
                reference_tokens,
                self.input_ids,
                self.ref_input_ids,
                self.sep_idx,
                self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
            lig.summarize()
            self.attributions_list.append(lig)

    def _run(
        self,
        text: str,
        embedding_type: int = None,
    ) -> list:  # type: ignore
        if embedding_type is None:
            embeddings = self.word_embeddings
        else:
            if embedding_type == 0:
                embeddings = self.word_embeddings
            elif embedding_type == 1:
                if self.accepts_position_ids and self.position_embeddings is not None:
                    embeddings = self.position_embeddings
                else:
                    warnings.warn(
                        "This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            else:
                embeddings = self.word_embeddings

        self.text = self._clean_text(text)

        self._calculate_attributions(embeddings=embeddings)
        return self.word_attributions  # type: ignore

    def __call__(
        self,
        text: str,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        """
        Calculates attribution for `text` using the model
        and tokenizer given in the constructor.
        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.
        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.
        Args:
            text (str): Text to provide attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        self.attributions_list = []

        return self._run(text, embedding_type=embedding_type)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s