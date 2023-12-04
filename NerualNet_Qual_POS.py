import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch import nn
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy
nlp = spacy.load("en_core_web_sm")
'''
Trains the the  dataset
'''
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

'''
Evaluates the results
'''
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


'''
Mddel for training text classification
'''
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    '''
    Intializes weights
    '''
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    '''
    Forward propagation of the model
    '''
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

'''
Removes stopwords
'''
def remove_stopwords(text):
    text = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word.isalpha() and not word in stop_words]
    return ''.join(text)

'''

'''
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

'''

'''
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1




#Setups data
df = pd.read_csv('C:/Users/socce/PycharmProjects/NeuralNet/lingustic_quality_inter.csv')
df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)
#df['static_text'] = df.apply(lambda row: remove_stopwords(row['static_text']), axis=1)

df = df.drop(['log_level'], axis=1).drop(['lemmas'], axis=1).drop(['log_messages'], axis=1).drop(['tag'], axis=1).drop(['dep'], axis=1).drop(['alpha'], axis=1).drop(['stop'], axis=1)
print(df.head())

#column_names = ['label', 'log_messages']
#df['log_level'] = df['log_level'].replace('info', 1.0).replace('error', 2.0).replace('warn', 3.0)
#df = df.reindex(columns=column_names)
print(df.head())
df_test = df.sample(frac = 0.3)
df_train = df.drop(df_test.index)

#Preping the pipeline
tokenizer = get_tokenizer("basic_english")
train_iter = list(df_train.itertuples(index=False, name=None))

##Creates the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

'''
A method a sentence into a set of numeric tokens
'''
text_pipeline = lambda x: vocab(tokenizer(x))

'''
Makes offsets the labels by 1 to drop them to base 0
'''
label_pipeline = lambda x: int(x) #- 1

#Use a cuda device if one is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter = list(df_train.itertuples(index=False, name=None))
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

train_iter = list(df.itertuples(index=False, name=None))
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 6
model = TextClassificationModel(vocab_size= vocab_size, embed_dim= emsize,num_class= num_class).to(device)



# Hyperparameters
EPOCHS = 300  # epoch
LR = 1  # learning rate
BATCH_SIZE = 1  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.01)
total_accu = None
train_iter, test_iter = list(df_train.itertuples(index=False, name=None)), list(df_test.itertuples(index=False, name=None))
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))
