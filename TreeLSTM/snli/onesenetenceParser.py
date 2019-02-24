import os
import time
import glob

import torch
from torch import optim
from torch import nn

from torchtext import data
from torchtext import datasets

from parser_predictor import Parser

from util import get_args

def calcu_loss(trans_truths,trans_pres):
    trans_len = trans_truths.size(0)
    trans_loss, trans_acc = 0, 0
    for n_trans in range(trans_len):
        trans_truth = trans_truths[n_trans]
        trans_pre = trans_pres[n_trans]
        # first is the max value in dim 1, and the second is the index of the maximum
        trans_preds_max = trans_pre.max(1)[1]
        trans_loss += criterion(trans_pre, trans_truth)
        trans_acc += torch.sum((trans_preds_max.data == trans_truth.data).float())
    return trans_loss,trans_acc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
is_training = True
#--------------setting device-------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
args = get_args()
torch.cuda.set_device(0)
LOG_FOUT = open(os.path.join(args.save_path, 'log_train.txt'), 'w')
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
LOG_FOUT.write(str(header) + '\n')
LOG_FOUT.flush()
#--------------prepare training test and vali data-------------
inputs = datasets.nli.ParsedTextField(lower=args.lower)
transitions = datasets.nli.ShiftReduceField()
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)

print(train[0].__dict__.keys())
print(train[0].premise[:10])
print(train[0].premise_transitions[:10])
#inputs.build_vocab(train, dev, test, max_size=100000, vectors=vec)
inputs.build_vocab(train, dev, test)
answers.build_vocab(train)
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, device=device)#######

#--------------------------- -------------

#--------------configure of setting-------------
config = args
config.n_embed = len(inputs.vocab)
config.lr = 2e-3 # 3e-4
config.lr_decay_by = 0.75
config.lr_decay_every = 1 #0.6
config.regularization = 0 #3e-6
config.mlp_dropout = 0.07
config.embed_dropout = 0.08 # 0.17
config.n_mlp_layers = 2
config.d_tracker = 64
config.d_mlp = 1024
config.d_hidden = 300
config.d_embed = 300
config.d_proj = 600
torch.backends.cudnn.enabled = False
#------------------initialize of the model--------- -------------
model = Parser(config)
model.cuda()
criterion = nn.CrossEntropyLoss()
opt = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.9, eps=1e-6,
                weight_decay=config.regularization)
#----------------------------------------
iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
print(header)
LOG_FOUT = open(os.path.join(args.save_path, 'log_train.txt'), 'w')
LOG_FOUT.write(str(header) + '\n')
LOG_FOUT.flush()
#----------------------------------------
if is_training:
    for epoch in range(args.epochs):
        LOG_FOUT.write('----'*10+ str(epoch) + '----'*10+ '\n')
        LOG_FOUT.flush()
        train_iter.init_epoch()#Set up the batch generator for a new epoch.
        n_correct = n_total = train_loss = 0
        for batch_idx, batch in enumerate(train_iter):
            iterations += 1
            model.train()
            opt.zero_grad()
            trans_pred = model(batch)
            trans = batch.premise_transitions
            sentenceLEN = trans.size(0)
            loss,acc = calcu_loss(trans,trans_pred)
            n_correct += acc
            n_total += batch.batch_size * sentenceLEN
            train_acc = 100. * n_correct / n_total
            loss.backward();
            opt.step();
            train_loss += loss.data.item() * batch.batch_size
            #show or save or evaluate the result of the training
            if iterations % 1 == 0:#1000 args.save_every
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, train_loss / n_total, iterations)
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
            if iterations % args.dev_every == 0:#1000
                model.eval(); dev_iter.init_epoch()
                n_dev_correct = dev_loss = n_total_dev = 0
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    trans_pred_dev = model(dev_batch)
                    trans_dev = dev_batch.premise_transitions
                    sentenceLEN_dev = trans_dev.size(0)
                    loss_dev, acc_dev = calcu_loss(trans_dev, trans_pred_dev)
                    n_dev_correct += acc_dev
                    dev_loss += loss_dev.data.item()  * dev_batch.batch_size
                    n_total_dev += dev_batch.batch_size * sentenceLEN_dev
                    #data has only one dim
                dev_acc = 100. * n_dev_correct / n_total_dev
                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx,len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), train_loss / n_total, dev_loss / len(dev), train_acc, dev_acc))
                n_correct = n_total = train_loss = 0
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                    snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss / len(dev), iterations)
                    torch.save(model.state_dict(), snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)
            elif iterations % args.log_every == 0:#50
                wout = (log_template.format(time.time() - start,
                    epoch, iterations, 1 + batch_idx, len(train_iter),
                    100. * (1 + batch_idx) / len(train_iter), train_loss / n_total, ' ' * 8,
                    n_correct / n_total, ' ' * 12))
                print(wout)
                n_correct = n_total = train_loss = 0
                LOG_FOUT.write(str(wout) + '\n')
                LOG_FOUT.flush()
    LOG_FOUT.close()
else:# using for test
    model_path = os.path.join(args.save_path, 'best_snapshot')
    model_name = model_path + '_1.pt'
    model.load_state_dict(torch.load(model_name))
    trans_pred = model(batch)




