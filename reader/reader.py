from typing import Optional, List, Tuple
import pdb
from collections import namedtuple, defaultdict
import shutil

from util.utils import Alphabet


GOOGLE_NEWS_WORD2VEC_FILE = "./embeddings/GoogleNews-vectors-negative300.bin"
PUBMED_WORD2VEC_FILE = "./embeddings/PubMed-shuffle-win-2.bin"

SentInst = namedtuple('SentInst', 'tokens chars entities')

PREDIFINE_TOKEN_IDS = {'DEFAULT': 0}
PREDIFINE_CHAR_IDS = {'DEFAULT': 0, 'BOT': 1, 'EOT': 2}


class Reader:
    def __init__(self) -> None:

        self.label_alphabet: Optional[Alphabet] = None

        self.train: Optional[List[SentInst]] = None
        self.dev: Optional[List[SentInst]] = None
        self.test: Optional[List[SentInst]] = None

    @staticmethod
    def _read_file(filename: str, mode: str = 'train') -> List[SentInst]:
        sent_list = []
        max_len = 0
        num_thresh = 0
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "":  # last few blank lines
                    break

                raw_tokens = line.split(' ')
                tokens = raw_tokens
                chars = [list(t) for t in raw_tokens]

                entities = next(f).strip()
                if entities == "":  # no entities
                    sent_inst = SentInst(tokens, chars, [])
                else:
                    entity_list = []
                    entities = entities.split("|")
                    for item in entities:
                        pointers, label = item.split()
                        pointers = pointers.split(",")
                        if int(pointers[1]) > len(tokens):
                            pdb.set_trace()
                        span_len = int(pointers[1]) - int(pointers[0])
                        assert (span_len > 0)
                        if span_len > max_len:
                            max_len = span_len
                        if span_len > 6:
                            num_thresh += 1

                        new_entity = (int(pointers[0]), int(pointers[1]), label)
                        # may be duplicate entities in some datasets
                        if (mode == 'train' and new_entity not in entity_list) or (mode != 'train'):
                            entity_list.append(new_entity)

                    # assert len(entity_list) == len(set(entity_list)) # check duplicate
                    sent_inst = SentInst(tokens, chars, entity_list)
                assert next(f).strip() == ""  # separating line

                sent_list.append(sent_inst)
        print("Max length: {}".format(max_len))
        print("Threshold 6: {}".format(num_thresh))
        return sent_list

    def _gen_dic(self) -> None:
        label_set = set()

        for sent_list in [self.train, self.dev, self.test]:
            num_mention = 0
            for sentInst in sent_list:
                for entity in sentInst.entities:
                    label_set.add(entity[2])
                num_mention += len(sentInst.entities)
            print("# mentions: {}".format(num_mention))

        self.label_alphabet = Alphabet(label_set, 0)

    @staticmethod
    def _pad_batches(token_batches: List[List[List[str]]]) -> List[List[List[bool]]]:

        mask_batches = []

        for token_batch in token_batches:

            batch_len = len(token_batch)
            max_sent_len = len(token_batch[0])

            mask_batch = []

            for i in range(batch_len):

                sent_len = len(token_batch[i])

                mask = [True] * sent_len + [False] * (max_sent_len - sent_len)

                mask_batch.append(mask)

            mask_batches.append(mask_batch)

        return mask_batches

    def to_batch(self, batch_size: int) -> Tuple:
        ret_list = []

        for sent_list in [self.train, self.dev, self.test]:
            token_dic = defaultdict(list)
            label_dic = defaultdict(list)

            this_token_batches = []
            this_label_batches = []

            for sentInst in sent_list:

                token_vec = [t for t in sentInst.tokens]

                label_list = [(u[0], u[1], self.label_alphabet.get_index(u[2])) for u in sentInst.entities]

                token_dic[len(sentInst.tokens)].append(token_vec)
                label_dic[len(sentInst.tokens)].append(label_list)

            token_batches = []
            label_batches = []
            for length in sorted(token_dic.keys(), reverse=True):
                token_batches.extend(token_dic[length])
                label_batches.extend(label_dic[length])

            [this_token_batches.append(token_batches[i:i + batch_size])
             for i in range(0, len(token_batches), batch_size)]
            [this_label_batches.append(label_batches[i:i + batch_size])
             for i in range(0, len(label_batches), batch_size)]

            this_mask_batches = self._pad_batches(this_token_batches)

            ret_list.append((this_token_batches, this_label_batches, this_mask_batches))

        return tuple(ret_list)

    def read_all_data(self, file_path: str, train_file: str, dev_file: str, test_file: str) -> None:
        self.train = self._read_file(file_path + train_file)
        self.dev = self._read_file(file_path + dev_file, mode='dev')
        self.test = self._read_file(file_path + test_file, mode='test')
        self._gen_dic()

    @staticmethod
    def gen_vectors_google_news_word2vec(embed_path: str) -> None:
        shutil.copyfile(GOOGLE_NEWS_WORD2VEC_FILE, embed_path)

    @staticmethod
    def gen_vectors_pubmed_word2vec(embed_path: str) -> None:
        shutil.copyfile(PUBMED_WORD2VEC_FILE, embed_path)

    def debug_single_sample(self, label_list: List[Tuple[int, int, int]]) -> None:
        for label in label_list:
            print(label[0], label[1], self.label_alphabet.get_instance(label[2]))
