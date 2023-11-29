from datasets import load_dataset, list_datasets

datasets = list_datasets()



from pprint import pprint

print(f"🤩 Currently {len(datasets)} datasets are available on the hub:")
pprint(datasets, compact=True)



dataset_ = load_dataset('cnn_dailymail', '3.0.0', split='train[:15]')


print(dataset_)



print(f"👉Dataset len(dataset): {len(dataset_)}")
print("\n👉First item 'dataset[0]':")
pprint(dataset_[0])


# Importing librareis
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl

from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader



import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    """Custom dataset class for text summarization using PyTorch DataLoader.

    For more information about Dataset and DataLoader, see:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, texts, summaries, tokenizer, source_len, summ_len):
        """
        Initialize the Dataset.

        Args:
            texts (list): List of input texts.
            summaries (list): List of target summaries.
            tokenizer: Tokenizer for text encoding.
            source_len (int): Maximum length for input text.
            summ_len (int): Maximum length for target summary.
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return len(self.summaries) - 1

    def __getitem__(self, index):
        """
        Get a single data sample from the dataset.

        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            Tuple containing:
            - source input IDs
            - source attention mask
            - target input IDs
            - target attention mask
        """
        text = ' '.join(str(self.texts[index]).split())
        summary = ' '.join(str(self.summaries[index]).split())

        # Article text pre-processing
        source = self.tokenizer.batch_encode_plus([text],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')
        # Summary Target pre-processing
        target = self.tokenizer.batch_encode_plus([summary],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')

        return (
            source['input_ids'].squeeze(),
            source['attention_mask'].squeeze(),
            target['input_ids'].squeeze(),
            target['attention_mask'].squeeze()
        )

class BARTDataLoader(pl.LightningDataModule):
    '''Pytorch Lightning Model Dataloader class for BART'''

    def __init__(self, tokenizer, text_len, summarized_len, file_path,
                 corpus_size, columns_name, train_split_size, batch_size):
        """
        Initialize the BARTDataLoader.

        Args:
            tokenizer: Tokenizer for text encoding.
            text_len (int): Maximum length for input text.
            summarized_len (int): Maximum length for target summary.
            file_path (str): Path to the CSV data file.
            corpus_size (int): Number of rows to read from the CSV file.
            columns_name (list): List of column names to use.
            train_split_size (float): Size of the training split (e.g., 0.8 for 80%).
            batch_size (int): Batch size for data loading.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.summarized_len = summarized_len
        self.input_text_length = summarized_len
        self.file_path = file_path
        self.nrows = corpus_size
        self.columns = columns_name
        self.train_split_size = train_split_size
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Load and preprocess the data from the CSV file.
        """
        data = pd.read_csv(self.file_path, nrows=self.nrows, encoding='latin-1')
        data = data[self.columns]
        data.iloc[:, 1] = 'summarize: ' + data.iloc[:, 1]
        self.text = list(data.iloc[:, 0].values)
        self.summary = list(data.iloc[:, 1].values)

    def setup(self, stage=None):
        """
        Split the data into training and validation sets.

        Args:
            stage (str): The current stage ('fit' or 'test').
        """
        X_train, y_train, X_val, y_val = train_test_split(
            self.text, self.summary, train_size=self.train_split_size
        )

        self.train_dataset = (X_train, y_train)
        self.val_dataset = (X_val, y_val)

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.
        """
        train_data = Dataset(texts=self.train_dataset[0],
                             summaries=self.train_dataset[1],
                             tokenizer=self.tokenizer,
                             source_len=self.text_len,
                             summ_len=self.summarized_len)
        return DataLoader(train_data, self.batch_size)

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.
        """
        val_dataset = Dataset(texts=self.val_dataset[0],
                              summaries=self.val_dataset[1],
                              tokenizer=self.tokenizer,
                              source_len=self.text_len,
                              summ_len=self.summarized_len)
        return DataLoader(val_dataset, self.batch_size)




import torch
import pytorch_lightning as pl
from transformers import AdamW

class AbstractiveSummarizationBARTFineTuning(pl.LightningModule):
    """Abstractive summarization model class for fine-tuning BART."""

    def __init__(self, model, tokenizer):
        """
        Initialize the AbstractiveSummarizationBARTFineTuning model.

        Args:
            model: Pre-trained BART model.
            tokenizer: BART tokenizer.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                decoder_attention_mask=None, lm_labels=None):
        """
        Forward pass for the model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for input.
            decoder_input_ids: Target token IDs.
            decoder_attention_mask: Attention mask for target.
            lm_labels: Language modeling labels.

        Returns:
            Model outputs.
        """
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=decoder_input_ids
        )

        return outputs

    def preprocess_batch(self, batch):
        """
        Reformat and preprocess the batch for model input.

        Args:
            batch: Batch of data.

        Returns:
            Formatted input and target data.
        """
        input_ids, source_attention_mask, decoder_input_ids, \
        decoder_attention_mask = batch

        y = decoder_input_ids
        decoder_ids = decoder_input_ids
        source_ids = input_ids
        source_mask = source_attention_mask

        return source_ids, source_mask, decoder_ids, decoder_attention_mask, decoder_attention_mask

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the batch.

        Returns:
            Loss for the training step.
        """
        input_ids, source_attention_mask, decoder_input_ids, \
        decoder_attention_mask, lm_labels = self.preprocess_batch(batch)

        outputs = self.forward(input_ids=input_ids, attention_mask=source_attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               lm_labels=lm_labels
                       )
        loss = outputs.loss

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the batch.

        Returns:
            Loss for the validation step.
        """
        input_ids, source_attention_mask, decoder_input_ids, \
        decoder_attention_mask, lm_labels = self.preprocess_batch(batch)

        outputs = self.forward(input_ids=input_ids, attention_mask=source_attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               lm_labels=lm_labels
                       )
        loss = outputs.loss

        return loss

    def training_epoch_end(self, outputs):
        """
        Calculate and log the average training loss for the epoch.

        Args:
            outputs: List of training step outputs.
        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('Epoch', self.trainer.current_epoch)
        self.log('avg_epoch_loss', {'train': avg_loss})

    def val_epoch_end(self, loss):
        """
        Calculate and log the average validation loss for the epoch.

        Args:
            loss: List of validation step losses.
        """
        avg_loss = torch.stack([x["loss"] for x in loss]).mean()
        self.log('avg_epoch_loss', {'Val': avg_loss})

    def configure_optimizers(self):
        """
        Configure and return the optimizer for the model.

        Returns:
            Optimizer for training.
        """
        model = self.model
        optimizer = AdamW(model.parameters())
        self.opt = optimizer

        return [optimizer]



# Tokenizer
# Upload the curated_data_subset.csv if using Colab or change the path to a local file
model_ = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Dataloader
# Initialize a DataLoader for processing and loading data
dataloader = BARTDataLoader(tokenizer=tokenizer, text_len=512,
                            summarized_len=150,
                            file_path='curated_data_subset.csv',
                            corpus_size=50, columns_name=['article_content','summary'],
                            train_split_size=0.8, batch_size=2)

# Read and pre-process data
dataloader.prepare_data()

# Train-test Split
# Split the data into training and validation sets
dataloader.setup()




# Main Model class
# Create an instance of the AbstractiveSummarizationBARTFineTuning model
model = AbstractiveSummarizationBARTFineTuning(model=model_, tokenizer=tokenizer)




# Trainer Class
# Initialize a PyTorch Lightning Trainer for training and evaluation
# You can specify the number of GPUs (e.g., gpus=1) if available, or remove it if not.
trainer = pl.Trainer(check_val_every_n_epoch=1, max_epochs=5)

# Fit model
# Train the model using the specified trainer and data loader
trainer.fit(model, dataloader)



from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig




def summarize_article(article):
    # Load BART model and tokenizer
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and encode the article
    inputs = tokenizer.encode(article, return_tensors='pt', max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


article = '''The team working to rescue the 41 workers trapped inside the Silkyara-Barkot tunnel in Uttarakhand’s Uttarkashi is just five metres away from the men, Chief Minister Pushkar Singh Dhami has said. He added that the rescuers have dug through 52 metres of debris. The rescue operations are expected to reach their destination today. Rat-hole mining commenced Monday evening as rescue efforts to reach the 41 workers trapped inside the Silkyara tunnel entered its 16th day. At least 6 rat-hole miners arrived from Delhi and Jhansi to enter the tunnels and dig manually. The trapped workers are behind 60 metres of debris and rescuers are roughly 12 metres away. Additional Secretary, Union Ministry of Road Transport and Highways, Mahmood Ahmed said that simultaneously, work of vertical drilling is going on at a fast pace and they have prepared around 36 metres of the vertical tunnel. A total of 86 metres have to be drilled vertically to prepare an escape passage, with pipes of 1.2 metres in diameter being laid through the top of the tunnel, work on which began on Sunday as a second option after the auger machine was damaged. The blades of the auger machine stuck in the rubble at Silkyara tunnel were removed early on Monday.'''


summary = summarize_article(article)
print("Summary:")
print(summary)
