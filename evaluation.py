
# coding: utf-8

# In[1]:

import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from itertools import ifilter
from random import randint
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import numpy as np
import torch, pdb, csv


# In[2]:

BATCH_SIZE = 130000
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, hid_dim, pretrained=None):
        super(Word2Vec, self).__init__()
        self.hid_dim = hid_dim
        
        #these are by intent to learn separate embedding matrices, we return word_emb
        self.word_emb = nn.Embedding(vocab_size, hid_dim)
        if pretrained is not None:
            self.word_emb.weight.data.copy_(pretrained)
        
        self.context_emb = nn.Embedding(vocab_size, hid_dim)
        if pretrained is not None:
            self.context_emb.weight.data.copy_(pretrained)        
        
        self.sigmoid = nn.LogSigmoid()
        
    def forward(self, wrd, cntxt, labels):
        wrd_vec = self.word_emb(wrd) # N * 1 * D
        cntxt_vec = self.context_emb(cntxt) # N * 5 * D
        res = torch.bmm(wrd_vec, cntxt_vec.view(BATCH_SIZE, self.hid_dim, -1))
        res = res.squeeze(1)
        res = res * labels
        res = self.sigmoid(res)
        
        # these are N * (1 + neg_exmpl) logsigmoid values
        # for each mini-batch we have a probability score for the 5 contexts
        # return res
        
        return (torch.sum(res)*-1.0)/res.size(0)


# In[3]:

def get_sim(wrd_indx, k, mat, descending=True):
    vec = mat[wrd_indx, :].unsqueeze(1)
    othrs = torch.mm(mat, vec)
    othrs, ind = torch.sort(othrs, 0, descending)
    topk = ind[:k]
    results = []
    for i in range(topk.size()[0]):
        results.append(topk[i][0])
    return results

def get_score(wrd1, wrd2, mat, word2index):
    return torch.dot(mat[word2index[wrd1],:], mat[word2index[wrd2], :])

def get_glovedict(glove_path):
    vocab_d = set()
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word = word.strip().lower()
            vocab_d.add(word)
            
    return vocab_d
    
def get_gloveready(glove_path, vocab_size, dim, word2index):
    pretrained_weight = torch.FloatTensor(vocab_size, dim)
    fnd = 0
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word = word.strip().lower()
            if word in word2index:
                ind = word2index[word]
                pretrained_weight[ind, :] = torch.from_numpy(np.array(list(map(float, vec.split()))))
                fnd += 1

    print('Found {0} words with glove vectors, total was {1}'.format(fnd, vocab_size))
    return pretrained_weight

def process_lines(data):
    pairs, vocab = set(), {}
    for cn, l in enumerate(data):
        dt = l.split("|||")
        score = float(dt[3].split(" ")[1].split("=")[1])
        if score < 3.4:
            continue
        wrd1, wrd2 = dt[1], dt[2]
        wrd1, wrd2 = wrd1.strip(), wrd2.strip()

        if ".pdf" not in wrd1 and ".pdf" not in wrd2 and wrd1.isalpha() and wrd2.isalpha():
            sc = editdist_score(wrd1, wrd2)
            if sc > min(len(wrd1), len(wrd2))/2:
                if wrd1 + " " + wrd2 not in pairs and wrd2 + " " + wrd1 not in pairs:
                    pairs.add(wrd1 + " " + wrd2)
                    if wrd1 not in vocab:
                        vocab[wrd1] = 1
                    else:
                        vocab[wrd1] += 1

                    if wrd2 not in vocab:
                        vocab[wrd2] = 1
                    else:
                        vocab[wrd2] += 1

    return pairs, vocab

def get_vocab(min_freq, flName=None, lines=None):
    if flName is not None:
        with open(flName) as fp:
            lines = fp.readlines()
    
    return process_lines(lines)

def get_chunks(lines, cn):
    chunks = []
    chunk_size = len(lines)//cn
    for i in range(0, chunk_size*cn + 1):
        chunk = lines[i*chunk_size:i*chunk_size + chunk_size]
        chunks.append(chunk)
    return chunks

def editdist_score(p1, p2):
    n, m = len(p1), len(p2)
    dp = [[0 for x in range(m+1)] for x in range(n+1)]

    for i in range(n+1):
        for j in range(m+1):
            if i == 0:
                dp[0][j] = j
            elif j == 0:
                dp[i][0] = i            
            elif p1[i-1] == p2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[n][m]

def filter_data(pairs, word2index):
    new_pairs = set()
    fp = open("ppdb-processed.txt", "w")
    for line in pairs:        
        p1, p2 = line.split(" ")
        if p1 in word2index and p2 in word2index:
            new_pairs.add(p1 + " " + p2)
            fp.write(line)
            fp.write("\n")
            
    fp.close()
    return new_pairs


# In[4]:

def load_opposites():
    oppos = {}
    with open('sneha_antonyms.csv', 'r') as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            if row[1] == 'antonym' and float(row[2]) > 0.85:
                wrd1, wrd2 = row[3], row[4]
                wrd1 = wrd1.strip().lower()
                wrd2 = wrd2.strip().lower()
                
                if wrd1 not in oppos:
                    oppos[wrd1] = []
                oppos[wrd1].append(wrd2)
                if wrd2 not in oppos:
                    oppos[wrd2] = []
                oppos[wrd2].append(wrd1)
    return oppos

def init_train(glove_path, dim, min_count, neg_exmpl):
    g_vocab = get_glovedict(glove_path)
    pairs, tok_freq = get_vocab(min_count, flName="ppdb-2.0-l-lexical")
    opposites = load_opposites()
    for wrd in opposites:
        if wrd not in tok_freq:
            tok_freq[wrd] = 0
        tok_freq[wrd] += 1
        
        antos = opposites[wrd]
        for awrd in antos:
            if awrd not in tok_freq:
                tok_freq[awrd] = 0
            tok_freq[awrd] += 1
            
    vocab = set(tok_freq.keys())
    vocab = vocab.intersection(g_vocab)

    word2index, index2word = {}, {}
    
    for wrd in vocab:
        if tok_freq[wrd] >= min_count:
            index2word[len(index2word)] = wrd
            word2index[wrd] = len(index2word) - 1
        else:
            tok_freq[wrd] = 0
            
    pairs = filter_data(pairs, word2index)
    print("Data ready: {} {} {}".format(len(index2word), len(pairs), len(vocab)))
    return pairs, word2index, index2word, vocab, tok_freq, opposites


# In[5]:

glove_path, dim, min_count, neg_exmpl = "glove.6B.50d.txt", 50, 1, 10
pairs, word2index, index2word, vocab, tok_freq, opposites = init_train(glove_path, dim, min_count, neg_exmpl)    


# In[6]:

def read_data(file, k):
    file = open(file,'r')
    lines = file.readlines()
    lines.pop(0)
    examples = []
    for i in lines:
        i = i.strip()
        i = i.lower()
        if len(i) > 0:
            i = i.split()
            ex = (i[0], i[1], float(i[k]))
            examples.append(ex)
    return examples

def getCorrelation(lines, We, word2index):
    gold, pred, skip = [], [], 0
    for i in lines:
        if i[0] not in word2index or i[1] not in word2index:
            skip += 1
            continue            
        vec1 = We[word2index[i[0]], :].cpu().numpy()
        vec2 = We[word2index[i[1]], :].cpu().numpy()
        pred.append(-1*cosine(vec1,vec2)+1)
        gold.append(i[2])
    print("Processed: {} total size was: {}".format(len(lines) - skip, len(lines)))
    return (spearmanr(pred,gold)[0])


# In[7]:

pretrained_weight = get_gloveready(glove_path, len(word2index), dim, word2index)
pretrained_weight = torch.nn.functional.normalize(pretrained_weight)
botk = {}
for wrd in vocab:
    res = get_sim(word2index[wrd], 10, pretrained_weight, False)
    botk[wrd] = res


# In[8]:

for w in botk['smashed']:
    print index2word[w],

print("")    
for w in get_sim(word2index['smashed'], 50, pretrained_weight):
    print index2word[w],


# In[9]:

mdl = Word2Vec(len(index2word), dim)    
mdl.load_state_dict(torch.load('./mdl_skipgme7.pth', map_location=lambda storage, loc: storage))
w2vmat = mdl.word_emb.weight.data.cpu()
wnorm = torch.norm(w2vmat, 2, 1, True)
w2vmat = w2vmat/wnorm


# In[26]:

get_score("big", "shrink", w2vmat, word2index)


# In[11]:

simlex_lines = read_data('./SimLex-999/SimLex-999.txt', 3)


# In[12]:

simlex_lines[0]


# In[13]:

getCorrelation(simlex_lines, w2vmat, word2index)


# In[14]:

get_score("small", "shrink", w2vmat, word2index)


# In[15]:

for w in get_sim(word2index['eat'], 10, w2vmat):
    print index2word[w], 


# In[16]:

getCorrelation(simlex_lines, pretrained_weight, word2index)


# In[17]:

get_score("big", "shrink", pretrained_weight, word2index)


# In[18]:

ws353sim_lines = read_data('./wordsim353/wordsim_simg.txt', 2)
ws353rel_lines = read_data('./wordsim353/wordsim_relg.txt', 2)


# In[19]:

ws353sim_lines[0]


# In[20]:

getCorrelation(ws353sim_lines, w2vmat, word2index)


# In[21]:

getCorrelation(ws353rel_lines, w2vmat, word2index)


# In[22]:

getCorrelation(ws353sim_lines, pretrained_weight, word2index)


# In[23]:

getCorrelation(ws353rel_lines, pretrained_weight, word2index)


# In[24]:

for w in get_sim(word2index['fear'], 10, w2vmat):
    print index2word[w], 


# In[18]:

get_sim('eat', 10, w2vmat, word2index)


# In[29]:

get_sim('eat', 10, pretrained_weight, word2index)


# In[76]:

get_sim('fear', 10, pretrained_weight, word2index)


# In[77]:

get_sim('fear', 10, w2vmat, word2index)


# In[78]:

get_sim('like', 10, w2vmat, word2index)


# In[79]:

get_sim('like', 10, pretrained_weight, word2index)


# In[ ]:



