# begin script -------------------------------------------------------

""" models.py
module holding FineTunedQA, our OOP approach to multi-choice qa,
via fine-tuning the encoder of UnifiedQA (khashabi et al. 2020), on
SocialIQA (Sap et al. 2019). we define multi-choice qa as a binary
classification problem, where we take in one candidate answer at
time, get its raw logits, and then softmax of the choices to extract
the highest scoring answer. 
"""

__author__ = 'Christopher Garcia Cordova'

# imports ------------------------------------------------------------

from   data_processing import SocialIQA
from   torch           import nn
import torch
from   tqdm            import tqdm

# class def ----------------------------------------------------------

class FineTunedQA(nn.Module):

    """
    class for building, training, and testing a fine-tuned UnifiedQA
    model, for multi-choice question answering. this model treats
    this task as a binary classification problem, wheren
    each candidate answer is classified as either being the correct
    answer or not. thus, dataset objs and loaders should be
    configured accordingly.
    """

    def __init__(self, uni_qa_encoder, encoder_hid_dim=512,
                     num_potential_answers=3, device='cpu'):

        """
        class constructor.
            params:
                uni_qa_encoder: type: pt.Model | hf.Model:
                    the encoder block of unified qa, producing
                    contextualized sentence embeddings for
                    classification. importantly, the forward pass
                    of this encoder should return the last hidden
                    representation of each token, as the first
                    element in the returned tuple.
                encoder_hid_dim: type: int:
                    optional
                    default val, 512.
                    specify the size of the contextualized vectors.
                num_potential_answers: type: int:
                    optional
                    default val, 3.
                    specify how many multiple choice answers per
                    question.
                device: type: str:
                    optional
                    default val, cpu.
                    specify available gpus.
            return: type: None.
        """

        # init with the parent class.
        super().__init__()

        # record encoder to current instance. 
        self.encoder = uni_qa_encoder

        # init a linear layer. 
        self.linear  =\
            nn.Linear(
                encoder_hid_dim,
                1
            )

        # record potential answers and device.
        self.num_candidates = num_potential_answers
        self.device         = device

    def forward(self, inputs, mask):
        
        """
        defines the forward pass of all instances.
            params:
                inputs: type: pt.tensor:
                    provide inputs encoded with appropriate
                    tokenizer.
                mask: type: pt.tensor:
                    provide a boolean, mask tensor, indicating
                    which ids are pads which are not.
            return: type: pt.tensor:
                    returns unnormalied logits, raw.
        """

        # run forward on encoder to get contextualized sent
        # embeddings.
        outputs = self.encoder(inputs, mask)

        # get the last_hidden state rep of the encoder on input.
        # [batch_size, max_src_len, enc_hid_dim]. we also make
        # sure we set pad tok outputs to zero, so they don't
        # influence our reduction to get the sent rep in the
        # next statement.
        last_hidden_states =\
            outputs[0] * mask.unsqueeze(-1)

        # we want to pool hidden states to get a represenation
        # for the entire sentence, either sum or mean. we do so
        # along the row space. reduction:
        # [batch_size, enc_hid_dim]
        sentence_embs =\
            torch.sum(
                last_hidden_states,
                dim=-2
            )

        # pass the extracted sentence embeddings to the linear
        # layer of the model to get logits. transform:
        # [batch_size, 1]
        logits = self.linear(sentence_embs)

        return logits 

    def tune(self, batches, optimizer, criterion, epochs=3,
                 valid=None):

        """
        method for tuning the current model on the multi-choice
        question-answer target task.
            params:
                data: type: pt.DataLoader:
                    a data loader for generating samples, where
                    each batch is a matrix of size:
                    [batch_size * num_candidates_answers, 
                     max_seq_len]. this is because the model
                    will perform binary classification to try
                    and predict which choice is the correct
                    answer for each candidate paired with the
                    question and context. a batch should be
                    a tuple containing encoded inputs, the mask
                    thereof, as well as binary vector of targets.
                optimizer: type: pt.optim.
                    provide what to optimize with.
                criterion: type: pt.nn.loss:
                    a loss function to optimize on.
                epochs: type: int:
                    optional
                    default val, 3.
                    specify the number of epochs to train for.
                valid: type: pt.Dataset:
                    optional
                    default val, None.
                    provide a validation set to track performance
                    on per epoch of training. 
            return: type: None.
        """

        # record accuracy of model on validation to perform early
        # stopping.
        prev_accuracy = 0
        curr_accuracy = 0

        # how many candidate answers there are, used for computing
        # loss and accuracy correctly.
        num_choices = self.num_potential_answers

        # specify that model is in training.
        self.train()

        # enter training.
        print('Fine-tuning model:')
        for epoch in range(1, epochs+1):
            # for recording avg loss per epoch.
            epoch_loss = 0

            # train on this dataset.
            print('Epoch '+str(epoch)+' out of '+str(epochs)+':')
            for batch in tqdm(batches):
                # we interate over each candidate answer type.
                # forget all previous grad comps if any.
                optimizer.zero_grad()

                # extract inputs, mask, targets from current batch.
                inputs, mask, targets =\
                    (x.to(self.device) for x in batch)

                # get the forward pass.
                logits = self(inputs, mask)

                # raw logits are of size [batch_size, 1] while
                # targets is a vector of lenghth batch_size,
                # so we flatten logits to be of the same shape
                # vector after sigmoid activation. 
                activations =\
                    torch.sigmoid( logits ).view(1, -1).squeeze()

                # compute loss.
                loss = criterion(activations, targets) 

                # now backprop, computing gradients.
                loss.backward()

                # now take a step in the direction of the
                # gradients.
                optimizer.step()

                # record loss.
                epoch_loss += loss.item()

            # compute avg loss for this epoch.
            print('Loss:', epoch_loss / (len(batches) * num_choices))

            # check whether to validate model during training, if
            # dev set was provided. 
            if valid: curr_accuracy = self.infer(valid) 

            # if model performance improved for this epoch,
            # save model, to enforce early stopping.
            if prev_accuracy < curr_accuracy:
                prev_accuracy = curr_accuracy
                self.save()
                print('Model saved at: ./fine-tuned-uni-qa.pt')

        # after all is done with training, set model to eval mode.
        self.eval()

    def infer(self, batches, data=None):

        """
        method for running inference on model, this method
        assumes test data is supervised.
            params:
                batches: type: pt.DataLoader:
                    a data loader for generating batches from the
                    provided dev set.
            return: type: float:
                    the accuracy on the given dataset.
        """

        # record accuracy.
        total_correct = 0
        total_samples = 0

        print('Validating:')
        with torch.no_grad():
            for batch in tqdm(batches):
                # extract inputs, mask, targets from current batch.
                inputs, mask, targets =\
                    (x.to(self.device) for x in batch)

                # run a forward pass on the model, producing raw
                # logits. we reshape to a matrix of size
                # [batch size, num canidate answers], so we
                # can softmax over the colspace at each row
                # to get most probable candidate answer.
                batch_size  = len(targets) // self.num_candidates
                num_choices = self.num_candidates

                logits =\
                    self(
                        inputs, mask 
                    ).view(batch_size, num_choices)

                targets =\
                    targets.view(batch_size, num_choices)

                # having reshaped to a matrix allows us to softmax
                # over candidate answer scores, along the colspace.
                probs = torch.softmax( logits, dim=-1 )

                # now we can retrive the argmaxes along the col
                # space for each row to extract what the model thinks
                # is the most probable answer to each question. 
                preds = probs.argmax( dim=-1 )

                # now we can compute accuracy, seeing how many
                # times the model picked the right answer.
                preds =\
                    (
                        targets[row][pred].item()
                        for row, pred in enumerate(preds)
                    )

                total_correct += sum(preds)
                total_samples += batch_size

            acc = total_correct / total_samples
            print('Accuracy:', acc)

        return acc

    def save(self, path='./fine-tuned-uni-qa.pt'):

        """
        method for saving the current model instance.
            params:
                path: type: str:
                    optional
                    default val, './fine-tuned-uni-qa.pt'.
                    specify the relative or absolute path for where
                    to save the model.
            return: type: None.
        """

        torch.save(self.state_dict(), path)
            
# end script ---------------------------------------------------------
