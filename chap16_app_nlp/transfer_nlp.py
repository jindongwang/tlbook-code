import torch
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--nclass', type=int, default=4)
    parser.add_argument('--no-pretrain', action='store_true')
    args = parser.parse_args()
    return args

def load_data():
    dataset = load_dataset('tweet_eval', 'emotion')
    return dataset

def tokenizer(data):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tok_data = data.map(tokenize_function, batched=True)
    return tok_data

def load_model(pretrain=True):
    if pretrain:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.nclass)
    else:
        from transformers import BertConfig, BertForSequenceClassification
        config = BertConfig.from_pretrained(args.model, num_labels=args.nclass)
        model =  BertForSequenceClassification(config)
    return model

def train(model, optimizer, loaders):
    num_epochs = args.nepochs
    num_training_steps = num_epochs * len(loaders['train'])
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    best_acc = 0
    for e in range(num_epochs):
        model.train()
        for batch in loaders['train']:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        eval_acc = eval(model, loaders['eval'])
        print(f'Epoch: [{e}/{num_epochs}] loss: {loss:.4f}, eval_acc: {eval_acc:.4f}')
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), 'bestmodel.pkl')
    # final test on test loader
    test_acc = eval(model, loaders['test'], 'bestmodel.pkl')
    print(f'Test accuracy: {test_acc:.4f}')

def eval(model, dataloader, model_path=None):
    metric = load_metric('accuracy')
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    res = metric.compute()
    return res['accuracy']

def post_process(tok_data):
    tok_data = tok_data.remove_columns(["text"])
    tok_data = tok_data.rename_column("label", "labels")
    tok_data.set_format("torch")
    
    loader_tr = DataLoader(tok_data["train"], shuffle=True, batch_size=args.batchsize)
    loader_eval = DataLoader(tok_data["validation"], batch_size=args.batchsize)
    loader_te = DataLoader(tok_data['test'], batch_size=args.batchsize)
    loaders = {"train": loader_tr, 'eval': loader_eval, 'test': loader_te}
    return loaders

if __name__ == "__main__":
    args = get_args()
    dataset = load_data()
    tok_data = tokenizer(dataset)
    print(tok_data)
    loaders = post_process(tok_data)
    
    model = load_model(pretrain=not args.no_pretrain)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(model, optimizer, loaders)
