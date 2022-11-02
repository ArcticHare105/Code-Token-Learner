import os
import random
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
import torch
from anytree import AnyNode, RenderTree
from anytree import find

def get_token(node):
    token = ''

    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token

def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))

def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)

def getnodes(node,nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child,nodelist)

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)

def getnodeid_depthfirst(node, nodeidindex):
    nodeidindex.append(node.id)
    for child in node.children:
        getnodeid_depthfirst(child, nodeidindex)

def getnodeid_breadthfirst(node, nodeidindex, is_root=True):
    if is_root:
        nodeidindex.append(node.id)
    for child in node.children:
        nodeidindex.append(child.id)
    for child in node.children:
        getnodeid_breadthfirst(child, nodeidindex, is_root=False)        

def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)

def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)

def countnodes(node,ifcount,whilecount,forcount,blockcount):
    token=node.token
    if token=='IfStatement':
        ifcount+=1
    if token=='WhileStatement':
        whilecount+=1
    if token=='ForStatement':
        forcount+=1
    if token=='BlockStatement':
        blockcount+=1
    print(ifcount,whilecount,forcount,blockcount)
    for child in node.children:
        countnodes(child,ifcount,whilecount,forcount,blockcount)

def createast():
    asts=[]
    paths=[]
    alltokens=[]
    dirname = 'BCB/bigclonebenchdata/'
    for rt, dirs, files in os.walk(dirname):
        for file in files:
            programfile=open(os.path.join(rt,file),encoding='utf-8')
            #print(os.path.join(rt,file))
            programtext=programfile.read()
            #programtext=programtext.replace('\r','')
            programtokens=javalang.tokenizer.tokenize(programtext)
            #print(list(programtokens))
            parser=javalang.parse.Parser(programtokens)
            programast=parser.parse_member_declaration()
            paths.append(os.path.join(rt,file))
            asts.append(programast)
            get_sequence(programast,alltokens)
            programfile.close()
            #print(programast)
            #print(alltokens)
    astdict=dict(zip(paths,asts))
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount = 0
    switchcount = 0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    print(ifcount,whilecount,forcount,blockcount,docount,switchcount)
    print('allnodes ',len(alltokens))
    alltokens=list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    return astdict,vocabsize,vocabdict

def createseparategraph(astdict, vocablen, vocabdict, device):
    pathlist=[]
    treelist=[]

    print(len(astdict))
    for path,tree in astdict.items():

        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)

        x = []
        edgesrc = []
        edgetgt = []

        getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)

        depth_first_id = []
        breadth_first_id = []
        getnodeid_depthfirst(newtree, depth_first_id)
        getnodeid_breadthfirst(newtree, breadth_first_id, True)        

        edge_index=[edgesrc, edgetgt]
        astlength=len(x)

        pathlist.append(path)
        treelist.append([[x, edge_index], astlength])
        astdict[path]=[[x, edge_index], astlength, depth_first_id, breadth_first_id]

    return astdict

def creategmndata(id,treedict,vocablen,vocabdict,device):
    indexdir='BCB/'
    if id=='0':
        trainfile = open(indexdir+'traindata.txt')
        validfile = open(indexdir+'devdata.txt')
        testfile = open(indexdir+'testdata.txt')
    elif id=='11':
        trainfile = open(indexdir+'traindata11.txt')
        validfile = open(indexdir+'devdata.txt')
        testfile = open(indexdir+'testdata.txt')
    else:
        print('file not exist')
        quit()
    trainlist=trainfile.readlines()
    validlist=validfile.readlines()
    testlist=testfile.readlines()
    traindata=[]
    validdata=[]
    testdata=[]
    print('train data')
    traindata=createpairdata(treedict,trainlist,device=device)
    print('valid data')
    validdata=createpairdata(treedict,validlist,device=device)
    print('test data')
    testdata=createpairdata(treedict,testlist,device=device)
    return traindata, validdata, testdata

def createpairdata(treedict,pathlist,device):
    datalist=[]
    countlines=1
    for line in pathlist:
        #print(countlines)
        countlines += 1
        pairinfo = line.split()
        code1path='BCB'+pairinfo[0].strip('.')
        code2path='BCB'+pairinfo[1].strip('.')
        label=int(pairinfo[2])
        data1 = treedict[code1path]
        data2 = treedict[code2path]
        x1, edge_index1, ast1length, df1, bf1 = data1[0][0],data1[0][1], data1[1], data1[2], data1[3]
        x2, edge_index2, ast2length, df2, bf2 = data2[0][0],data2[0][1], data2[1], data2[2], data2[3]

        data = [[x1, x2, edge_index1, edge_index2, df1, df2, bf1, bf2], label]
        datalist.append(data)
    return datalist

if __name__ == '__main__':
    astdict, vocabsize, vocabdict=createast()
    treedict=createseparategraph(astdict, vocabsize, vocabdict, device='cpu')
