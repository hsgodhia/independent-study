import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from itertools import ifilter
from random import randint
from joblib import Parallel, delayed

BATCH_SIZE = 100
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, hid_dim, pretrained=None):
        super(Word2Vec, self).__init__()
        self.hid_dim = hid_dim
        
        #these are by intent to learn separate embedding matrices, we return word_emb
        self.word_emb = nn.Embedding(vocab_size, hid_dim)
        if pretrained is not None:
            self.word_emb.weight.data.copy_(pretrained)
        
        self.context_emb = nn.Embedding(vocab_size, hid_dim)
        #if pretrained is not None:
        #    self.context_emb.weight.data.copy_(pretrained)        
        
        self.sigmoid = nn.LogSigmoid()
        
    def forward(self, wrd, cntxt, labels):
        wrd_vec = self.word_emb(wrd) # N * 1 * D
        cntxt_vec = self.context_emb(cntxt) # N * 5 * D
        cntxt_vec = torch.transpose(cntxt_vec, 1, 2)
        res = torch.bmm(wrd_vec, cntxt_vec)
        res = res.squeeze(1)
        res = res * labels
        res = self.sigmoid(res)
        
        # these are N * (1 + neg_exmpl) logsigmoid values
        # for each mini-batch we have a probability score for the 5 contexts
        # return res
        
        return (torch.sum(res)*-1.0)/res.size(0)
        
def negative_sampling_tbl(vocab, tok_freq, vocab_size, idx2word):
    total_cn = 0
    for wrd in vocab:
        total_cn += pow(tok_freq[wrd],0.75)
        
    tbl_size, wrd_idx = int(1e6), 0
    table = torch.LongTensor(tbl_size) # defaults to a column vector with only 1 dimension
    wrd_prob = pow(tok_freq[idx2word[wrd_idx]], 0.75)/total_cn
    
    for i in range(0, tbl_size):
        table[i] = wrd_idx
        ind = i*1.0
        if ind/tbl_size > wrd_prob:
            wrd_idx += 1
            wrd_prob += pow(tok_freq[idx2word[wrd_idx]], 0.75)/total_cn
        if wrd_idx >= vocab_size:
            wrd_idx -= 1

    return table
        
# return the sample context the first one being the true word and other being negative
def sample_context(table, neg_cn, cntxt):
    cntxts, i = [], 0
    cntxts.append(cntxt)
    while i < neg_cn:
        ind = randint(0, len(table) - 1)
        neg_ctx = table[ind]
        if neg_ctx != cntxt:
            cntxts.append(neg_ctx)
            i += 1
    return cntxts


def train_pair(wrd_idx, cntxts, labels, mdl, criterion, optimizer, index2word):
    """
        wrd_idx: is the input word which is predicting the context
        cntxts: contains 1 positive word idx's and remaining negative words idx's forming the context
    """
    loss = mdl(wrd_idx, cntxts, labels)
    #preds = mdl(wrd_idx, cntxts, labels)
    #loss = criterion(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.data[0]
    
def per_trainepoch(mdl, lines, table, criterion, optimizer, labels, word2index, index2word, neg_exmpl=20, win_size=5):
    
    track_loss, batch_wrd_idx, batch_cntxts = [], [], []
    batch_count = 0

    for k, l in enumerate(lines):
        l = l.strip()
        wrds = l.split(" ")
        for i, wrd in enumerate(wrds):
            wrd_idx = word2index[wrd]
            for j in range(max(0, i - win_size), min(len(wrds), i + win_size)):
                cntxt_wrd = wrds[j]
                if j != i:
                    cntxt_idx = word2index[cntxt_wrd]
                    cntxts = sample_context(table, neg_exmpl, cntxt_idx)
                    batch_wrd_idx.append(wrd_idx)
                    batch_cntxts.append(cntxts)

                    if len(batch_wrd_idx) == BATCH_SIZE:
                        batch_count += 1
                        var_wrd_idx = Variable(torch.LongTensor(batch_wrd_idx)).unsqueeze(1)
                        var_cntxts = Variable(torch.LongTensor(batch_cntxts))

                        if torch.cuda.is_available():
                            var_wrd_idx = var_wrd_idx.cuda()
                            var_cntxts = var_cntxts.cuda()

                        lossval = train_pair(var_wrd_idx, var_cntxts, labels, mdl, criterion, optimizer, index2word)
                        if k % 50000 == 0:
                            print("loss:{} line:{}".format(lossval, k))
                        track_loss.append(lossval)
                        batch_wrd_idx[:], batch_cntxts[:] = [], []
    
    print("tuples processed (wrd, cntxt):", batch_count*BATCH_SIZE)
    return sum(track_loss)/len(track_loss)

def get_sim(wrd, k, mat, word2index):
    if wrd not in word2index:
        return None
    vec = mat[word2index[wrd], :].unsqueeze(1)
    othrs = torch.mm(mat, vec)
    othrs, ind = torch.sort(othrs, 0, descending=True)
    topk = ind[:k]
    for i in range(topk.size()[0]):
        print(index2word[topk[i][0]])

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
        if score < 3.3:
            continue
        wrd1, wrd2 = dt[1], dt[2]
        wrd1, wrd2 = wrd1.strip(), wrd2.strip()

        if ".pdf" not in wrd1 and ".pdf" not in wrd2 and wrd1.isalpha() and wrd2.isalpha():
            sc = editdist_score(wrd1, wrd2)
            if sc > min(len(wrd1), len(wrd2))/2 + 2:
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

def main(EPOCHS):
    glove_path, dim, min_count, neg_exmpl = "glove.6B.50d.txt", 50, 1, 60
    g_vocab = get_glovedict(glove_path)
    pairs, tok_freq = get_vocab(min_count, flName="ppdb-2.0-l-lexical")

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
    vocab_size = len(index2word)
    #chunks = get_chunks(lines, 7)
    #retvals = Parallel(n_jobs=7)(delayed(process_line)([chunk, g_vocab]) for chunk in chunks)
    
    print("Data ready: {} {} {}".format(vocab_size, len(pairs), len(vocab)))
    pretrained_weight = get_gloveready(glove_path, vocab_size, dim, word2index)
    pretrained_weight = torch.nn.functional.normalize(pretrained_weight)
    print("Glove loaded")
    negative_tbl = negative_sampling_tbl(vocab, tok_freq, vocab_size, index2word)

    print("filtered data size:", len(pairs))
    
    #free memory
    del vocab, g_vocab, tok_freq
    mdl = Word2Vec(vocab_size, dim, pretrained_weight)    
    
    #init.xavier_normal(mdl.word_emb.weight)
    #init.xavier_normal(mdl.context_emb.weight)
    
    #free memory
    del pretrained_weight
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(mdl.parameters(), lr = 0.1)
    
    print('Training..')
    labels = Variable(torch.ones(BATCH_SIZE, 1 + neg_exmpl) * -1.0)
    labels[:, 0] = labels[:, 0] * -1.0
    
    if torch.cuda.is_available():
        labels = labels.cuda()
        negative_tbl = negative_tbl.cuda()
        mdl.cuda()

    for u in range(EPOCHS):
        loss_epoch = per_trainepoch(mdl, pairs, negative_tbl, criterion, optimizer, labels, word2index, index2word,neg_exmpl)
        print("---completed: {} and loss: {} ---".format(u, loss_epoch))
        torch.save(mdl.state_dict(), './mdl_skipgm.pth')
    
main(10)