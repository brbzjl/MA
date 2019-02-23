import os
import time
import numpy as np

import torch

sentence = ['the','church','has','cracks','in','the','ceiling','.']
transition = [1,1,2,1,1,1,1,1,2,2,2,2,1,2,2]

class node(object):
    def __init__(self, value, children = []):
        self.value = value
        self.children = children

    # def __str__(self, level=0):#dfs
    #     ret = "\t"*level+repr(self.value)+"\n"
    #     for child in self.children:
    #         ret += child.__repr__(level+1)
    #     return ret

    def __repr__(self, level=0):
        # ret = "\t"*level+repr(self.value)+"\n"
        # for child in self.children:
        #     ret += child.__repr__(level+1)
        return self.value

buffer =[]
stack =[]

for item in transition:
    if item == 1:#shift push the item into stack
        stack.append(sentence.pop(0))
    if item == 2:#reduce pop the first two elements in stack and reduce and push back to the buffer
        root = node('grandmother')
        root.children = [(stack.pop(-2)), (stack.pop(-1))]
        stack.append(root)

print(stack[0].value)
print(stack[0])

def dfs_show(root, depth):
    if depth == 0:
        print("root:" + root.value + "")
    if hasattr(root,'children'):
        for item in root.children:
            print("|      " * depth + "+--" , item)
            dfs_show(item, depth +1)

dfs_show(stack[0],0)