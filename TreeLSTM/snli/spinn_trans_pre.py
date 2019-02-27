import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import itertools


def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two or three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None):
        super(Reduce, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided as
        iterables and batched internally into tensors.

        Additionally augments each new node with pointers to its children.

        Args:
            left_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~autograd.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left`` and ``right``
                attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])
        out = unbundle(tree_lstm(left[1], right[1], lstm_in))
        # for o, l, r in zip(out, left_in, right_in):
        #     o.left, o.right = l, r
        return out


class Tracker(nn.Module):

    def __init__(self, size, tracker_size):
        super(Tracker, self).__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        self.transition = nn.Linear(tracker_size, 4)
        self.state_size = tracker_size
    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [Variable(
                x.data.new(x.size(0), self.state_size).zero_())]
        self.state = self.rnn(x, self.state)
        return unbundle(self.state), self.transition(self.state[0])

class SPINN1(nn.Module):

    def __init__(self, config):
        super(SPINN1, self).__init__()
        self.config = config
        assert config.d_hidden == config.d_proj / 2
        self.reduce = Reduce(config.d_hidden, config.d_tracker)
        self.tracker = Tracker(config.d_hidden, config.d_tracker) #300,64

    def forward(self, buffers, transitions):
        # squeeze all the dimensions of input of size 1 removed.
        # l = [torch.randn(1,3) for _ in range(3)]#list contains 3 1x3 tensor
        # inputs is a   43x128x600 split(inptus,1,1) will build a list seperated in dim 1
        xx = torch.split(buffers, 1, 1)  # xx is a list contains 128 51x1x300
        for b in xx:
            # 43x1x600 squeeze 43x600, list has 43 1x600 tensors
            bb = list(torch.split(b.squeeze(1), 1, 0))
        # 43x1x600 squeeze 43x300, list has  43 1x600 tensors,
        # buffers is a list with 128 lists , 43 1x600 tensor
        buffers = [list(torch.split(b.squeeze(1), 1, 0))
                   for b in torch.split(buffers, 1, 1)]
        # 128 lists 2 items, 1x600
        stacks = [[buf[0], buf[0]] for buf in buffers]

        self.tracker.reset_state()

        if transitions is not None:
            num_transitions = transitions.size(0)
            trans_loss, trans_acc = 0, 0
        else:
            num_transitions = 15#num_transitions = len(buffers[0]) * 2 - 3
        #num_transitions 83 128
        trans_preds_batch = []
        for i in range(num_transitions):

            if hasattr(self, 'tracker'):
                # list 81 1x128 tensor
                tracker_states, trans_preds = self.tracker(buffers, stacks) # 128x4 ,4 is the probable results
                if trans_preds is not None:
                    trans_preds_batch.append(trans_preds)
                    # if transitions is not None:
                    #     trans = transitions[i]  # grundtruth
                    # else:
                    #     trans = trans_preds.max(1)[1]
                    trans = trans_preds.max(1)[1]
                    # print('predict: ',trans)
                    # print('true trans: ',transitions[i])
                    # if transitions is not None:
                    #     trans_loss += F.cross_entropy(trans_preds, trans)
                    #     # a= (trans_preds_max.data == trans.data)
                    #     # b= (trans_preds_max.data == trans.data).float()
                    #     trans_acc += torch.sum((trans_preds_max.data == trans.data).float())
                    # else:
                    #     trans = trans_preds
            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)  # each element in zip has 128 items
            for transition, buf, stack, tracking in batch:  # 128 loops iterations
                # 128 batch size iterations
                if transition == 3:  # shift
                    if len(buf)>2:
                        stack.append(buf.pop())
                elif transition == 2:  # reduce
                    # 81 lists 1x600
                    if len(stack)>3:
                        rights.append(stack.pop())
                        lefts.append(stack.pop())
                        # 81 1x128
                        trackings.append(tracking)
                    # if here in each step calculate the reduced result? what will happen
                        # reduced is a tuple
                        reduced = self.reduce(lefts[-1].unsqueeze(0), rights[-1].unsqueeze(0), tracking.unsqueeze(0))
                        stack.append(reduced[0])
            # if rights:#if there are some items needed to be reduced, reduce them add push back into the corresponding stack
            #     reduced = iter(self.reduce(lefts, rights, trackings))
            #     for transition, stack in zip(trans.data, stacks):# 128 loops
            #         if transition == 2:
            #             stack.append(next(reduced))
        return trans_preds_batch
