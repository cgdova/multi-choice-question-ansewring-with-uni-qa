# begin script -------------------------------------------------------

""" data_processing.py
module defining class definitions for processing data for UnifiedQA,
a variant of T5, for format-agnostic question answering. 
in this version, we batch the data so that
candidates of a single question are contiguous, unlike previous
version of this script. 
"""

__author__ = 'Christopher Garcia Cordova'

# imports ------------------------------------------------------------

from   transformers     import AutoTokenizer
from   datasets         import load_dataset
from   torch.utils.data import Dataset, DataLoader
import torch

# class def ----------------------------------------------------------

class SocialIQA(Dataset):

    """
    class for processing and preparing data for finetuning
    UnifiedQA on the SocialIQA dataset, a multiple choice
    commonsense reasoning dataset.
    """

    def __init__(self, file_path, tokenizer, file_type='json',
                     split_name='train'):

        """
        class constructor.
            params:
                file_path: type: str:
                    the relative or absolute path of the file
                    holding the data.
                tokenizer: type: transformers.tokenizers:
                    the tokenizer with which to tokenizer, encode,
                    and decode data samples with.
                file_type: type: str:
                    optional
                    default val, json.
                    specify how the data is formatted.
                split_name: type: str:
                    optional
                    default val, train.
                    specify what part of the dataset split
                    that data is: train, dev, test.
            return: type: None.
        """

        # init from parent class.
        super().__init__()

        # load in the data from the local machine. returns
        # a dictionary of datasets objs.
        data =\
            load_dataset(
                file_type,
                data_files={ split_name : file_path }
            )

        # store data in current instance, we have to extract
        # it from dict.
        self.data = data[split_name]

        # record tokenizer.
        self.tokenizer = tokenizer

        # useful dict for encoding answers.
        self.answer2int = {'A': 0 , 'B': 1, 'C': 2}

    def __getitem__(self, idx):

        """
        returns the i-th data sample, where i = idx.
            params:
                idx: type: int:
                    specify the index of the i-th
                    data sample.
            return: type: tuple(list(str), str):
                    each candidate answer A, B, C as a list,
                    as well as the label for the correct one. 
        """

        # extract a sample from the data at idx.
        context, question, answerA, answerB, answerC, correct =\
                self.data[idx].values()

        # pre-prep data in format UnifiedQA expects. for each
        # candidate A, B, C.
        candidateA =\
            self._concat_candidate(
                question,
                '(A) ' + answerA,
                context
            )

        candidateB =\
            self._concat_candidate(
                question,
                '(B) ' + answerB,
                context
            )

        candidateC =\
            self._concat_candidate(
                question,
                '(C) ' + answerC,
                context
            )

        return [candidateA, candidateB, candidateC], correct

    def __len__(self):

        """
        returns the number of samples in this dataset.
            params: type: None.
            return: type: None.
        """

        return len(self.data)

    def _concat_candidate(self, *args):

        """
        returns the concatenation of a collection of strings,
        delimited by \\n.
            params:
                *args: type: str.
                    any number of strings wanted.
            returns: type: str.
        """

        return ' \n '.join(args)

    def _prep_batch(self, batch):

        """
        defines a collate_fn for preparing batches for this
        dataset. each is three times the specified size, taking
        into account candidate answers.
            params:
                batch: type: list(tuple(str)):
                    a list of length batch size, where
                    each element is a single sample, with
                    candidate answers A, B, C in list,
                    and the correct one.
            return: type: tuple(pt.tensor):
                    the encoded input ids, attention masks,
                    and target answers, in that order.
                    the batch combines all candidates
                    answers for all questions into a single
                    tensor, though candidates belonging to
                    a particular question are contiguous. this
                    is also true for the mask and targets.
        """

        # collect the candidates answers.
        candidates = list()
        for x in batch: candidates.extend(x[0])

        # collect collect answers. 
        targets = [x[1] for x in batch]

        # encoded collect candidates.
        encoded = self.encode(candidates)

        # now we collect the input_ids returned by tokenizer.
        inputs = encoded['input_ids']

        # as well as the attention masks.
        masks  = encoded['attention_mask']

        # finally, we have to build a binary vector indicating
        # which are the correct answers in the inputs vector.
        # format a one-hot accordingly.
        answers = torch.zeros(len(targets), 3)
        for i, answer in enumerate(targets):
            answers[
                i, self.answer2int[answer]
            ] = 1

        # we flatten the matrix into the desired binary vector.
        answers = answers.view(1, -1).squeeze()

        return inputs, masks, answers

    def encode(self, inputs):

        """
        returns an encode representation of data, as processed
        by he tokenizer passed at construction.
            params:
                inputs: type: str | list(str):
                    the inputs to encode.
            return: type: dict(str: pt.tensors).
        """

        return\
            self.tokenizer(
                inputs,
                return_tensors='pt',
                padding=True
            )

    def decode(self, inputs):

        """
        returns inverse of encode.
            params:
                inputs: type: pt.tensor:
                    the inputs to encode, possibly batched.
            return: type: list(str):
                    the tensors decode into their corresponding
                    strings.
        """

        return\
            self.tokenizer.batch_decode(
                inputs,
                skip_special_tokens=True
            )

    def get_loader(self, batch_size=4):

        """
        returns a torch data loader for batching.
            params:
                batch_size: type: int:
                    the size of each batch.
            return: type: pt.DataLoader:
                    a data loader for batching data, with a
                    special collate_fn.
        """

        return\
            DataLoader(
                self,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self._prep_batch, 
            )

# end script ---------------------------------------------------------
