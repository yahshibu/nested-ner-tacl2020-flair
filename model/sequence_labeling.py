__author__ = 'max'
__maintainer__ = 'takashi'

from typing import List, Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
import flair
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence, Token

from module.crf import ChainCRF4NestedNER
from module.variational_rnn import VarMaskedFastLSTM
from module.embeddings import WordEmbeddings, CharacterEmbeddings, BertEmbeddings, FlairEmbeddings


class NestedSequenceLabel:
    def __init__(self, start: int, end: int, label: Tensor, children: List) -> None:
        self.start = start
        self.end = end
        self.label = label
        self.children = children


class BiRecurrentConvCRF4NestedNER(nn.Module):
    def __init__(self, embed_path: str, char_embed: int, num_filters: int, label_size: int, hidden_size: int = 256,
                 layers: int = 1, word_dropout: float = 0.20, lstm_dropout: float = 0.50, fine_tune: bool = False) \
            -> None:
        super(BiRecurrentConvCRF4NestedNER, self).__init__()

        self.embeddings: StackedEmbeddings \
            = StackedEmbeddings([WordEmbeddings(embed_path, word_dropout=word_dropout),
                                 CharacterEmbeddings(char_embedding_dim=char_embed, hidden_size_char=num_filters),
                                 BertEmbeddings("bert-large-uncased",
                                                layers="-1" if fine_tune else "-1,-2,-3,-4,-5,-6,-7,-8",
                                                pooling_operation="mean"),
                                 FlairEmbeddings("news-forward"),
                                 FlairEmbeddings("news-backward")])
        # standard dropout
        self.dropout_out: nn.Dropout2d = nn.Dropout2d(p=lstm_dropout)

        self.rnn: VarMaskedFastLSTM = VarMaskedFastLSTM(self.embeddings.embedding_length, hidden_size,
                                                        num_layers=layers, batch_first=True, bidirectional=True,
                                                        dropout=(lstm_dropout, lstm_dropout))

        self.reset_parameters()

        self.all_crfs: List[ChainCRF4NestedNER] = []

        for label in range(label_size):
            crf = ChainCRF4NestedNER(hidden_size * 2, 1)
            self.all_crfs.append(crf)
            self.add_module('crf%d' % label, crf)

        self.b_id: int = 0
        self.i_id: int = 1
        self.e_id: int = 2
        self.s_id: int = 3
        self.o_id: int = 4
        self.eos_id: int = 5

        self.device: Optional[torch.device] = None
        self.cpu()

    def reset_parameters(self) -> None:
        for name, parameter in self.rnn.named_parameters():
            nn.init.constant_(parameter, 0.)
            if name.find('weight_ih') > 0:
                if name.startswith('cell0.weight_ih') or name.startswith('cell1.weight_ih'):
                    bound = (6. / (self.rnn.input_size + self.rnn.hidden_size)) ** 0.5
                else:
                    bound = (6. / ((2 * self.rnn.hidden_size) + self.rnn.hidden_size)) ** 0.5
                nn.init.uniform_(parameter, -bound, bound)
                parameter.data[:2, :, :] = 0.
                parameter.data[3:, :, :] = 0.
            if name.find('bias_hh') > 0:
                parameter.data[1, :] = 1.

    def cuda(self, device: int = 0) -> "BiRecurrentConvCRF4NestedNER":
        for _, module in self.named_children():  # type: str, nn.Module
            if not isinstance(module, StackedEmbeddings):
                module.cuda(device)
        self.device = torch.device(device)
        return self

    def cpu(self) -> "BiRecurrentConvCRF4NestedNER":
        for _, module in self.named_children():  # type: str, nn.Module
            if not isinstance(module, StackedEmbeddings):
                module.cpu()
        self.device = torch.device('cpu')
        return self

    def to(self, *args, **kwargs):
        raise NotImplementedError

    def _get_rnn_output(self, tokens: List[List[str]], mask: Tensor = None) -> Tensor:

        sentences = []
        for token in tokens:
            sentence = Sentence()
            [sentence.add_token(Token(t.replace('\xa0', ' '))) for t in token]
            sentences.append(sentence)

        self.embeddings.embed(sentences)

        lengths = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(self.embeddings.embedding_length * longest_token_sequence_in_batch,
                                                dtype=torch.float, device=flair.device)

        all_embs = list()
        for sentence in sentences:
            all_embs += [emb for token in sentence.tokens for emb in token.get_each_embedding()]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[:self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

            for token in sentence.tokens:
                token.clear_embeddings()

        # [batch, length, word_dim]
        input = torch.cat(all_embs) \
            .view((len(sentences), longest_token_sequence_in_batch, self.embeddings.embedding_length))

        if self.device != flair.device:
            if self.device != torch.device('cpu'):
                input = input.cuda(self.device)
            else:
                input = input.cpu()

        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask)

        # apply dropout for the output of rnn
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output

    def forward(self, tokens: List[List[str]],
                target: Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]], mask: Tensor) -> Tensor:
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(tokens, mask=mask)

        # [batch, length, num_label, num_label]
        batch, length, _ = output.size()

        loss = []

        for label, crf in enumerate(self.all_crfs):
            target_batch = torch.cat(tuple([target_each.label.unsqueeze(0) for target_each in target[label]]), dim=0)

            loss_batch, energy_batch = crf.loss(output, target_batch, mask=mask)

            calc_nests_loss = crf.nests_loss

            def forward_recursively(loss: Tensor, energy: Tensor, target: NestedSequenceLabel, offset: int) -> Tensor:
                nests_loss_list = []
                for child in target.children:
                    if child.end - child.start > 1:
                        nests_loss = calc_nests_loss(energy[child.start - offset:child.end - offset, :, :],
                                                     child.label)
                        nests_loss_list.append(forward_recursively(nests_loss,
                                                                   energy[child.start - offset:child.end - offset, :, :],
                                                                   child, child.start))
                return sum(nests_loss_list) + loss

            loss_each = []
            for i in range(batch):
                loss_each.append(forward_recursively(loss_batch[i], energy_batch[i], target[label][i], 0))

            loss.append(sum(loss_each))

        loss = sum(loss)

        return loss / batch

    def predict(self, tokens: List[List[str]], mask: Tensor) \
            -> Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]]:
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(tokens, mask=mask)

        batch, length, _ = output.size()

        preds = []

        for crf in self.all_crfs:
            preds_batch, energy_batch = crf.decode(output, mask=mask)

            b_id = self.b_id
            i_id = self.i_id
            e_id = self.e_id
            o_id = self.o_id
            eos_id = self.eos_id
            decode_nest = crf.decode_nest

            def predict_recursively(preds: Tensor, energy: Tensor, offset: int) -> NestedSequenceLabel:
                length = preds.size(0)
                nested_preds_list = []
                index = 0
                while index < length:
                    id = preds[index]
                    if id == eos_id:
                        break
                    if id != o_id:
                        if id == b_id:  # B-XXX
                            start_tmp = index
                            index += 1
                            if index == length:
                                break
                            id = preds[index]
                            while id == i_id:  # I-XXX
                                index += 1
                                if index == length:
                                    break
                                id = preds[index]
                            if id == e_id:  # E-XXX
                                end_tmp = index + 1
                                nested_preds = decode_nest(energy[start_tmp:end_tmp, :, :])
                                nested_preds_list.append(predict_recursively(nested_preds,
                                                                             energy[start_tmp:end_tmp, :, :],
                                                                             start_tmp + offset))
                    index += 1
                return NestedSequenceLabel(offset, length + offset, preds, nested_preds_list)

            preds_each = []
            for i in range(batch):
                preds_each.append(predict_recursively(preds_batch[i, :], energy_batch[i, :, :, :], 0))

            preds.append(preds_each)

        return preds
