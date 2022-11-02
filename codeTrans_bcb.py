import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
from createclone_bcb import createast, creategmndata, createseparategraph
import models

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--data_setting", default='0')
parser.add_argument("--batch_size", default=10)
parser.add_argument("--dropout", default=0.5)
parser.add_argument("--num_layers", default=3)
parser.add_argument("--inp_dim_num", default=128)
parser.add_argument("--num_heads", default=4)
parser.add_argument("--head_dim_num", default=16)
parser.add_argument("--mlp_dim_num", default=128)
parser.add_argument("--num_epochs", default=5)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0.1)
args = parser.parse_args()

device=torch.device('cuda:0')
astdict, vocablen, vocabdict = createast()
treedict=createseparategraph(astdict, vocablen, vocabdict, device)
traindata,validdata,testdata=creategmndata(args.data_setting, treedict, vocablen, vocabdict, device)
print(len(traindata))

inp_dim_num=args.inp_dim_num
num_heads=args.num_heads
head_dim_num=args.head_dim_num
dropout=args.dropout
num_layers=args.num_layers
model = models.CloneTrans(vocablen, inp_dim_num, num_heads, head_dim_num, dropout, num_layers, device=device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion=nn.CosineEmbeddingLoss()
criterion2=nn.MSELoss()

def create_batches(data):
    batches = [data[graph:graph+args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches

def test(dataset):
    #model.eval()
    count=0
    correct=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results=[]
    for data, label in dataset:
        label = torch.tensor(label, dtype=torch.float, device=device)
        x1, x2, edge_index1, edge_index2, df1, df2, bf1, bf2 = data

        x1=torch.tensor(x1, dtype=torch.long, device=device)
        x2=torch.tensor(x2, dtype=torch.long, device=device)

        edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)

        df1 = torch.tensor(df1, dtype=torch.long, device=device)
        df2 = torch.tensor(df2, dtype=torch.long, device=device)
        bf1 = torch.tensor(bf1, dtype=torch.long, device=device)
        bf2 = torch.tensor(bf2, dtype=torch.long, device=device)        

        data1=[x1, edge_index1, df1, bf1]
        data2=[x2, edge_index2, df2, bf2]
        prediction1, prediction2, _, _ = model(data1, data2)
        output = F.cosine_similarity(prediction1,prediction2)
        results.append(output.item())
        prediction = torch.sign(output).item()

        if prediction>args.threshold and label.item()==1:
            tp+=1
            #print('tp')
        if prediction<=args.threshold and label.item()==-1:
            tn+=1
            #print('tn')
        if prediction>args.threshold and label.item()==-1:
            fp+=1
            #print('fp')
        if prediction<=args.threshold and label.item()==1:
            fn+=1
            #print('fn')
    print(tp,tn,fp,fn)
    p=0.0
    r=0.0
    f1=0.0
    if tp+fp==0:
        print('precision is none')
        return
    p=tp/(tp+fp)
    if tp+fn==0:
        print('recall is none')
        return
    r=tp/(tp+fn)
    f1=2*p*r/(p+r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    return results

epochs = trange(args.num_epochs, leave=True, desc = "Epoch")
for epoch in epochs:
    print(epoch)
    batches=create_batches(traindata)
    totalloss=0.0
    main_index=0.0
    for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        optimizer.zero_grad()
        batchloss= 0
        for data, label in batch:
            label_t = torch.tensor(label, dtype=torch.float, device=device)
            x1, x2, edge_index1, edge_index2, df1, df2, bf1, bf2 = data
            x1=torch.tensor(x1, dtype=torch.long, device=device)
            x2=torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)

            df1 = torch.tensor(df1, dtype=torch.long, device=device)
            df2 = torch.tensor(df2, dtype=torch.long, device=device)
            bf1 = torch.tensor(bf1, dtype=torch.long, device=device)
            bf2 = torch.tensor(bf2, dtype=torch.long, device=device)        

            data1=[x1, edge_index1, df1, bf1]
            data2=[x2, edge_index2, df2, bf2]

            prediction1, prediction2, _, _ = model(data1, data2)
            cossim = F.cosine_similarity(prediction1,prediction2)
            batchloss=batchloss + criterion2(cossim, label_t)
        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        totalloss+=loss
        main_index = main_index + len(batch)
        loss=totalloss / main_index
        epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

    devresults=test(validdata)
    devfile=open('bcbresult/' + 'dev_epoch_'+str(epoch+1),mode='w')
    for res in devresults:
        devfile.write(str(res)+'\n')
    devfile.close()

    testresults = test(testdata)        
    resfile=open('bcbresult/' + 'test_epoch_'+str(epoch+1),mode='w')
    for res in testresults:
        resfile.write(str(res)+'\n')
    resfile.close()
