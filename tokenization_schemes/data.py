import pickle as pkl
import spacy
import string

train=pkl.load(open("/home/jm7432/train.pkl","rb"))
test=pkl.load(open("/home/jm7432/test.pkl","rb"))

train_split=20000

training=train[:train_split//2]+train[12500:12500+train_split//2]
train_data=[]
train_target=[]
for sample in training:
    train_data+=[sample['text']]
    if sample['sentiment']=='pos':
        train_target+=[1]
    else:
        train_target+=[0]

validation=train[train_split//2:12500]+train[12500+train_split//2:]
val_data=[]
val_target=[]
for sample in validation:
    val_data+=[sample['text']]
    if sample['sentiment']=='pos':
        val_target+=[1]
    else:
        val_target+=[0]

test_data=[]
test_target=[]
for sample in test:
    test_data+=[sample["text"]]
    if sample['sentiment']=='pos':
        test_target+=[1]
    else:
        test_target+=[0]
print(len(train_data))
print(len(val_data))
print(len(test_data))

tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation

def tokenize(sent):
  tokens = tokenizer(sent)
  return [token.text for token in tokens]

def tokenize_dataset(dataset):
    token_dataset = []
    # we are keeping track of all tokens in dataset 
    # in order to create vocabulary later
    all_tokens = []
    
    for sample in dataset:
        tokens = tokenize(sample)
        token_dataset.append(tokens)
        all_tokens += tokens

    return token_dataset, all_tokens

# val set tokens
print ("Tokenizing val data")
val_data_tokens, _ = tokenize_dataset(val_data)
pkl.dump(val_data_tokens, open("/home/jm7432/val_data_tokens_noPrep.p", "wb"))

# test set tokens
print ("Tokenizing test data")
test_data_tokens, _ = tokenize_dataset(test_data)
pkl.dump(test_data_tokens, open("/home/jm7432/test_data_tokens_noPrep.p", "wb"))

# train set tokens
print ("Tokenizing train data")
train_data_tokens, all_train_tokens = tokenize_dataset(train_data)
pkl.dump(train_data_tokens, open("/home/jm7432/train_data_tokens_noPrep.p", "wb"))
pkl.dump(all_train_tokens, open("/home/jm7432/all_train_tokens_noPrep.p", "wb"))
