import numpy as np
import torch, pdb, csv
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from itertools import ifilter
from random import randint
from joblib import Parallel, delayed
from torch.optim import lr_scheduler
from random import shuffle
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from IPython.core.debugger import set_trace


def get_glovedict(glove_path):
    vocab_d = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word = word.strip().lower()
            vocab_d[word] = np.array(list(map(float, vec.split())))
            
    return vocab_d

#sample input: 2.0	[NP]	this one	this dimension	1.0,2.0,2.0,2.0,3.0
def read_data(flName):    
    with open(flName, 'r') as fp:
        lines = fp.readlines()
    pairs, max_len = [], 0
    scores= []
    for l in lines:
        dt = l.split("\t")
        score = float(dt[0])        
        max_len = max(len(dt[2].split(" ")), len(dt[3].split(" ")), max_len)
        pairs.append((score, dt[2], dt[3]))
        scores.append(score)
    scores = np.array(scores)
    print(np.mean(scores), np.std(scores))
    return pairs, max_len

def get_data_split(num):
    flName = "./human-labeled-data/ppdb-sample.tsv"
    pairs, max_len = read_data(flName)
    print("processing ppdb pairs count:", len(pairs))
    indx = np.random.permutation(len(pairs))[:num]
    inv_idx = np.random.permutation(len(pairs))[num:]
    
    val_pairs = [pairs[indi] for indi in indx]
    train_pairs = [pairs[indi] for indi in inv_idx]
    
    return train_pairs, val_pairs

BATCH_SIZE = 100
class DAN(nn.Module):
    # glove dict is a numpy matrix
    def __init__(self):
        super(DAN, self).__init__()
        """
        self.word_emb = nn.Embedding(glove_tensor.size(0), hid_dim, requires_grad=False)
        self.word_emb.weight.data.copy_(glove_tensor)
        self.word_emb.weight.requires_grad = False
        """
        self.relu = nn.ReLU()        
        self.lin1 = nn.Linear(300, 300)
        
        self.batn1 = nn.BatchNorm1d(300, affine=False)
        self.batn2 = nn.BatchNorm1d(300, affine=False)
        self.batn3 = nn.BatchNorm1d(400, affine=False)
        self.batn4 = nn.BatchNorm1d(100, affine=False)
        
        self.lin2 = nn.Linear(300, 300)
        self.lin3 = nn.Linear(901, 400)
        self.lin4 = nn.Linear(400, 100)
        self.lin5 = nn.Linear(100, 1)
        
    def forward(self, x1, x2):
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        # assume that they are converted to embeddings outside of the model
        o1 = self.relu(self.batn2(self.lin2(self.relu(self.batn1(self.lin1(x1))))))
        o2 = self.relu(self.batn2(self.lin2(self.relu(self.batn1(self.lin1(x2))))))
        
        cor = torch.sum(o1 * o2, 1, keepdim=True)
        nxt = torch.cat([o1, o2, torch.abs(o1-o2), cor], 1)
        score = self.lin5(self.relu(self.batn4(self.lin4(self.relu(self.batn3(self.lin3(nxt)))))))
        return score

gdict = get_glovedict("glove.840B.300d.txt")
train_pairs, val_pairs = get_data_split(500)

def train(epochs):
    glove_path, dim = "glove.840B.300d.txt", 300
    dan = DAN()    
    criterion = nn.MSELoss()
    
    if torch.cuda.is_available():
        dan.cuda()
    
    optimizer = optim.Adagrad(dan.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        loss_vals, batchx1, batchx2, scores = [], [], [], []
        for i, p in enumerate(train_pairs):
            score, x1, x2 = p

            vec1, vec2 = torch.zeros(dim), torch.zeros(dim)
            
            x1s = x1.split(" ")
            for wrd in x1s:
                if wrd in gdict:
                    vec1 += torch.from_numpy(gdict[wrd]).float()
            if len(x1s) > 0:
                vec1 = vec1/len(x1s)
                
            x2s = x2.split(" ")
            for wrd in x2s:
                if wrd in gdict:
                    vec2 += torch.from_numpy(gdict[wrd]).float()            
            if len(x2s) > 0:
                vec2 = vec2/len(x2s)
                
            if torch.cuda.is_available():
                vec2 = vec2.unsqueeze(0)
                vec1 = vec1.unsqueeze(0)
                
            if len(batchx1) == BATCH_SIZE:
                x1 = torch.stack(batchx1, 0)
                x1 = Variable(x1.cuda())
                x2 = torch.stack(batchx2, 0)
                x2 = Variable(x2.cuda())
                
                score_t = Variable(torch.FloatTensor([scores]).cuda())
                
                prob = dan(x1, x2)
                loss = criterion(prob, score_t)
                loss_vals.append(loss.data[0])
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batchx1, batchx2, scores = [], [], []

            else:
                batchx2.append(vec2)
                batchx1.append(vec1)                
                scores.append(score)

        print("after epoch {} loss is {} ".format(epoch, sum(loss_vals)*1.0/len(loss_vals)))
    torch.save(dan.state_dict(), './dan.pth')
    return dan

def evaluate_pairs(dan, val_pairs, dim):
    model_results, actual_results = [], []
    batchx1, batchx2, scores = [], [], []
    for i, p in enumerate(val_pairs):
        score, x1, x2 = p
        vec1, vec2 = torch.zeros(dim), torch.zeros(dim)

        x1s = x1.split(" ")
        for wrd in x1s:
            if wrd in gdict:
                vec1 += torch.from_numpy(gdict[wrd]).float()
        if len(x1s) > 0:
            vec1 = vec1/len(x1s)

        x2s = x2.split(" ")
        for wrd in x2s:
            if wrd in gdict:
                vec2 += torch.from_numpy(gdict[wrd]).float()            
        if len(x2s) > 0:
            vec2 = vec2/len(x2s)

        if torch.cuda.is_available():
            vec2 = vec2.unsqueeze(0)
            vec1 = vec1.unsqueeze(0)

        if len(batchx1) == BATCH_SIZE:
            x1 = torch.stack(batchx1, 0)
            x1 = Variable(x1.cuda())
            x2 = torch.stack(batchx2, 0)
            x2 = Variable(x2.cuda())

            score_t = Variable(torch.FloatTensor([scores]).cuda())

            prob = dan(x1, x2)
            model_results.extend(prob.data.cpu().numpy().tolist())
            actual_results.extend(scores)

            batchx1 = []
            batchx2 = []
            scores = []
        else:
            batchx2.append(vec2)
            batchx1.append(vec1)                
            scores.append(score)
                
    return actual_results, model_results

def test(dan):
    flName, dim = "./human-labeled-data/wiki-sample.tsv", 300
    pairs, max_len = read_data(flName)
    actual_r, model_r = evaluate_pairs(dan, val_pairs, dim)
    return spearmanr(actual_r, model_r)[0]

dan = train(500)
print("the spearman corelation on held out test set is:",test(dan))