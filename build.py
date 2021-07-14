# being script -------------------------------------------------------

""" build.py
this script makes use of classes in our modules models and 
data_processing to fine-tune unified qa on the social iqa dataset.
"""

__author__ = 'Christopher Garcia Cordova'

# imports ------------------------------------------------------------

from data_processing     import *
from models              import *
from transformers        import T5EncoderModel, AdamW
from torch.utils.data    import DataLoader
from torch.nn            import BCELoss
from tqdm                import tqdm

# procedure ----------------------------------------------------------

# get device on current machine.
device =\
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init tokenizer for t5-small
tokenizer =\
    AutoTokenizer.from_pretrained(
        'allenai/unifiedqa-t5-base'
    )

# init train data for task.
train =\
    SocialIQA(
        file_path='./data/train',
        tokenizer=tokenizer,
        file_type='json',
        split_name='train'
    )

# init valid data for task. 
valid =\
    SocialIQA(
        file_path='./data/dev',
        tokenizer=tokenizer,
        file_type='json',
        split_name='dev'
    )

test =\
    SocialIQA(
        file_path='./data/test',
        tokenizer=tokenizer,
        file_type='json',
        split_name='test'
    )


# init encoder block of unified qa.
encoder =\
    T5EncoderModel.from_pretrained('allenai/unifiedqa-t5-base')

# init model, consider making pooling the sentence representation
# optional: self.pool=True, False; consider letting the user
# pick the index at which to access the sent rep from the encoder
# in return of its foward pass: self.output_idx=[-inf, inf].
model =\
    FineTunedQA(
        uni_qa_encoder=encoder,
        encoder_hid_dim=768,
        num_potential_answers=3,
        device=device
    ).to(device)


# init optimizer and loss, and get batch loader.
loss = BCELoss()
optimizer = AdamW(model.parameters(), lr=1e-4)

# tune model on target task, hopefully.
model.tune(
    train.get_loader(batch_size=16),
    optimizer,
    loss,
    epochs=20,
    valid=valid.get_loader(batch_size=4)
)

model.save('./fine-tuned-uni-qa.pt')

model.infer(test.get_loader(batch_size=16), test)

# end script ---------------------------------------------------------
