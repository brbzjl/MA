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
import json
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
class mydataset():
    def __init__(self,path,device,data_format='json'):
        self.test_ppid_dataset_path = path
        self.data_format = data_format
        self.device = device
    def write_dataset(self):
        data_format = self.data_format.lower()
        # dict_dataset = [
        #     {"id": "0", "question1": "When do you use シ instead of し?",
        #      "question2": "When do you use \"&\" instead of \"and\"?",
        #      "label": "0"},
        #     {"id": "1", "question1": "Where was Lincoln born?",
        #      "question2": "Which location was Abraham Lincoln born?",
        #      "label": "1"},
        #     {"id": "2", "question1": "What is 2+2",
        #      "question2": "2+2=?",
        #      "label": "1"},
        # ]
        sentence = {"premise": "the church has cracks in the ceilin ."}
        dict_dataset = [sentence]
        with open(self.test_ppid_dataset_path, "w") as test_ppid_dataset_file:
            for example in dict_dataset:
                if data_format == "json":
                    test_ppid_dataset_file.write(json.dumps(example) + "\n")
                else:
                    raise ValueError("Invalid format {}".format(data_format))

    def build_dateset(self):
        question_field = data.Field(sequential=True)

        # fields = {"question1": ("q1", question_field),
        #           "question2": ("q2", question_field),
        #           "label": ("label", label_field)}
        fields = {"premise": ("premise", question_field)}
        test = data.TabularDataset(
            path=self.test_ppid_dataset_path, format=self.data_format, fields=fields)
        print(test[0].__dict__.keys())
        print(test[0].premise)
        #print(dataset[1].premise)
        question_field.build_vocab(test)
        iter = data.BucketIterator(test, repeat = True, batch_size=128, device=self.device)
        return iter




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

#--------------setting device-------------
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
args = get_args()
torch.cuda.set_device(0)
#--------------configure of setting-------------
config = args
config.n_embed = 36990#len(inputs.vocab) 36990
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
config.is_training = True
torch.backends.cudnn.enabled = False
#--------------prepare training test and vali data-------------
if config.is_training:
    inputs = datasets.nli.ParsedTextField(lower=args.lower)
    transitions = datasets.nli.ShiftReduceField()
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)
    #
    # print(train[0].__dict__.keys())
    # print(train[0].premise[:10])
    # print(train[0].premise_transitions[:10])
    #inputs.build_vocab(train, dev, test, max_size=100000, vectors=vec)
    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=device)#######

#--------------------------- -------------
#------------------initialize of the model--------- -------------
model = Parser(config)
model.cuda()
criterion = nn.CrossEntropyLoss()
opt = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.9, eps=1e-6,
                weight_decay=config.regularization)

#----------------------------------------
if config.is_training:
    # ----------------------------------------
    iterations = 0
    start = time.time()
    best_dev_acc = -1
    train_iter.repeat = False
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    os.makedirs(args.save_path, exist_ok=True)
    print(header)
    LOG_FOUT = open(os.path.join(args.save_path, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(header) + '\n')
    LOG_FOUT.flush()
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
    model_path = os.path.join(args.save_path, '2.pt')
    #model_name = model_path + '_1.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval();
    sentence = ['the', 'church', 'has', 'cracks', 'in', 'the', 'ceiling', '.']
    # batch = {'premise':sentence}
    # batch = torch.Tensor(batch).cuda()
    datamy = mydataset('test.json',device)
    datamy.write_dataset()
    data_inter = datamy.build_dateset()
    data_inter.init_epoch()
    for batch_idx, batch in enumerate(data_inter):
        trans_pred = model(batch)
        trans_len = len(trans_pred)#trans_predbatch.premise_transitions.size(0)
        pre1=[]
        pre2 = []
        for n_trans in range(trans_len):
            #trans_truth = batch.premise_transitions[n_trans]
            trans_pre = trans_pred[n_trans]
            # first is the max value in dim 1, and the second is the index of the maximum
            trans_preds_max = trans_pre.max(1)[1]
            pre1.append(trans_preds_max[0])
            #pre2.append(trans_truth[0])
        #print(batch.premise[:,0])
        #print(batch.premise)
        #print(test[0].premise_transitions)
        #print(pre2)
        print(pre1)
        break
class node(object):
    def __init__(self, value, children = []):
        self.value = value
        self.children = children
    def __repr__(self, level=0):
        return self.value
buffer =[]
stack =[]

for i,item in enumerate(pre1):
    if item == 3:#shift push the item into stack
        if len(sentence)>1:
            stack.append(sentence.pop(0))
    if item == 2:#reduce pop the first two elements in stack and reduce and push back to the buffer
        root = node('grandmother')
        root.children = [(stack.pop(-2)), (stack.pop(-1))]
        stack.append(root)
print(stack[0])

def dfs_show(root, depth):
    if depth == 0:
        print("root:" + root.value + "")
    if hasattr(root,'children'):
        for item in root.children:
            print("|      " * depth + "+--" , item)
            dfs_show(item, depth +1)

dfs_show(stack[0],0)



