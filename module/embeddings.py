__copyright__ = 'Copyright (C) 2018 Zalando SE'
__license__ = 'MIT'

from abc import abstractmethod
from typing import List
import torch
import torch.nn as nn
import flair
import flair.embeddings
from flair.data import Sentence

from module.dropout import VarDropout4Bert, VarDropout4Flair


class WordEmbeddings(flair.embeddings.WordEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None, word_dropout: float = 0.05) -> None:
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        super(WordEmbeddings, self).__init__(embeddings=embeddings, field=field)

        self.static_embeddings = False

        self.word_dropout = word_dropout

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        with torch.no_grad():

            for i, sentence in enumerate(sentences):

                for token_idx, token in enumerate(sentence.tokens):

                    word = token.text.replace(' ', '\xa0')

                    word_embedding = self.get_cached_vec(word=word)

                    token.set_embedding(self.name, word_embedding)

            if self.word_dropout == 0. or not self.training:
                return sentences

            pre_allocated_zero_tensor = torch.zeros((self.embedding_length, ), dtype=torch.float, device=flair.device)

            for i, sentence in enumerate(sentences):

                m = pre_allocated_zero_tensor.new_empty((len(sentence.tokens), )).bernoulli_(1. - self.word_dropout)

                for token_idx, token in enumerate(sentence.tokens):

                    if m[token_idx] == 0.:
                        token.set_embedding(self.name, pre_allocated_zero_tensor)

            return sentences


class CharacterEmbeddings(flair.embeddings.CharacterEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(self, path_to_char_dict: str = None, char_embedding_dim: int = 25, hidden_size_char: int = 25) -> None:
        """Uses the default character dictionary if none provided."""
        super(CharacterEmbeddings, self).__init__(path_to_char_dict=path_to_char_dict,
                                                  char_embedding_dim=char_embedding_dim,
                                                  hidden_size_char=hidden_size_char)

        nn.init.constant_(self.char_embedding.weight, 0.)
        for name, parameter in self.char_rnn.named_parameters():
            nn.init.constant_(parameter, 0.)
            if name.startswith('weight_ih'):
                if name == 'weight_ih_l0' or name == 'weight_ih_l0_reverse':
                    bound = (6. / (self.char_rnn.input_size + self.char_rnn.hidden_size)) ** 0.5
                else:
                    bound = (6. / ((2 * self.char_rnn.hidden_size) + self.char_rnn.hidden_size)) ** 0.5
                nn.init.uniform_(parameter, -bound, bound)
                parameter.data[:2 * self.char_rnn.hidden_size, :] = 0.
                parameter.data[3 * self.char_rnn.hidden_size:, :] = 0.
            if name.startswith('bias_hh'):
                parameter.data[self.char_rnn.hidden_size:2 * self.char_rnn.hidden_size] = 1.


class BertEmbeddings(flair.embeddings.BertEmbeddings):
    def __init__(self, bert_model_or_path: str = "bert-base-uncased", layers: str = "-1,-2,-3,-4",
                 pooling_operation: str = "first", fine_tune: bool = False, use_scalar_mix: bool = False) -> None:
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first')
        """
        super(BertEmbeddings, self).__init__(bert_model_or_path=bert_model_or_path, layers=layers,
                                             pooling_operation=pooling_operation, use_scalar_mix=use_scalar_mix)

        self.fine_tune = fine_tune
        self.static_embeddings = False

        self.model.embeddings.dropout = VarDropout4Bert(self.model.embeddings.dropout.p)
        for l in range(len(self.model.encoder.layer)):
            self.model.encoder.layer[l].attention.output.dropout \
                = VarDropout4Bert(self.model.encoder.layer[l].attention.output.dropout.p)
            self.model.encoder.layer[l].output.dropout \
                = VarDropout4Bert(self.model.encoder.layer[l].output.dropout.p)

        if fine_tune:
            self.model.embeddings.word_embeddings.weight.requires_grad = False
            self.model.embeddings.position_embeddings.weight.requires_grad = False
            self.model.embeddings.token_type_embeddings.weight.requires_grad = False
        else:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = False

        self.model.to(flair.device)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        with torch.set_grad_enabled(self.fine_tune and torch.is_grad_enabled()):

            # first, find longest sentence in batch
            longest_sentence_in_batch = len(
                max(
                    [
                        self.tokenizer.tokenize(sentence.to_tokenized_string())
                        for sentence in sentences
                    ],
                    key=len,
                )
            )

            # prepare id maps for BERT model
            features = self._convert_sentences_to_features(
                sentences, longest_sentence_in_batch
            )
            all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
                flair.device
            )
            all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
                flair.device
            )

            # put encoded batch through BERT model to get all hidden states of all encoder layers
            all_encoder_layers = self.model(all_input_ids, attention_mask=all_input_masks)[
                -1
            ]

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = all_encoder_layers[int(layer_index)][
                            sentence_index
                        ]
                        all_layers.append(layer_output[token_index])

                    if self.use_scalar_mix:
                        sm = flair.embeddings.ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

            return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return super(BertEmbeddings, self).embedding_length


class FlairEmbeddings(flair.embeddings.FlairEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(self, model: str, fine_tune: bool = False, chars_per_chunk: int = 512) -> None:
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'
                depending on which character language model is desired.
        :param fine_tune: if set to True, the gradient will propagate into the language model. This dramatically slows
                down training and often leads to overfitting, so use with caution.
        :param  chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster
                but requires ore memory. Lower means slower but less memory.
        """
        super(FlairEmbeddings, self).__init__(model=model, fine_tune=fine_tune, chars_per_chunk=chars_per_chunk)

        self.static_embeddings = False

        self.lm.drop = VarDropout4Flair(self.lm.drop.p)

        if fine_tune:
            self.lm.encoder.weight.requires_grad = False
        else:
            for name, parameter in self.lm.named_parameters():
                parameter.requires_grad = False

        self.lm.to(flair.device)

    def train(self, mode=True):

        # make compatible with serialized models (TODO: remove)
        if "fine_tune" not in self.__dict__:
            self.fine_tune = False
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

        return super(flair.embeddings.FlairEmbeddings, self).train(mode)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        with torch.set_grad_enabled(self.fine_tune and torch.is_grad_enabled()):

            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = [sentence.to_tokenized_string() for sentence in sentences]

            start_marker = "\n"
            end_marker = " "

            # get hidden states from language model
            all_hidden_states_in_lm = self.lm.get_representation(
                text_sentences, start_marker, end_marker, self.chars_per_chunk
            )

            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string()

                offset_forward: int = len(start_marker)
                offset_backward: int = len(sentence_text) + len(start_marker)

                for token in sentence.tokens:

                    offset_forward += len(token.text)

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    # if self.tokenized_lm or token.whitespace_after:
                    offset_forward += 1
                    offset_backward -= 1

                    offset_backward -= len(token.text)

                    # only clone if optimization mode is 'gpu'
                    if flair.embedding_storage_mode == "gpu":
                        embedding = embedding.clone()

                    token.set_embedding(self.name, embedding)

            del all_hidden_states_in_lm

        return sentences
