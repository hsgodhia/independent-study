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
        if pretrained is not None:
            self.context_emb.weight.data.copy_(pretrained)        
        
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

"""
will return the closest k words in the vocabulary so
as to ignore them when negative sampling
"""
def load_topk(k, vocab, word2index, pretrained_weight):
    botk = {}
    for wrd in vocab:
        botk[wrd] = get_sim(word2index[wrd], k, pretrained_weight, True)

    return botk

# return the sample context the first one being the true word and other being negative
def sample_context(table, neg_cn, cntxt, opposites, word2index, index2word, botk):
    cntxts, i, anto = [], 0, []
    cntxts.append(cntxt)
    
    
    if index2word[cntxt] in opposites:
        anto = opposites[index2word[cntxt]]
        for wrd in anto:
            if wrd in word2index:
                if len(cntxts) > neg_cn:
                    break
                cntxts.append(word2index[wrd])
    """
    # sample is not completely random but out of 40, 20 of them come from the bottomk
    for bwrd_id in botk[index2word[cntxt]]:
        if len(cntxts) > neg_cn:
            break
        cntxts.append(bwrd_id)
    """
    # instead of random sampling from the table we can also do distance sorting because we have pretrained word embeddings
    while len(cntxts) <= neg_cn:
        ind = randint(0, len(table) - 1)
        neg_ctx = table[ind]
        
        # do negative sampling by ensuring that negative words are not the top 100 similar or related words to current word
        if neg_ctx != cntxt and neg_ctx not in botk[index2word[cntxt]]:
            cntxts.append(neg_ctx)

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
    
def per_trainepoch(mdl, lines, table, criterion, optimizer, labels, word2index, index2word, botk, neg_exmpl, opposites, win_size=5):
    
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
                    cntxts = sample_context(table, neg_exmpl, cntxt_idx, opposites, word2index, index2word, botk)
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

def get_sim(wrd_indx, k, mat, descending=True):
    vec = mat[wrd_indx, :].unsqueeze(1)
    othrs = torch.mm(mat, vec)
    othrs, ind = torch.sort(othrs, 0, descending)
    topk = ind[:k]
    results = []
    for i in range(topk.size()[0]):
        results.append(topk[i][0])
    return results

def get_glovedict(glove_path):
    vocab_d = set()
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word = word.strip().lower()
            vocab_d.add(word)
            
    return vocab_d

def adjust_learning_rate(optimizer, epoch, interval, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
          
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

def main(EPOCHS):
    glove_path, dim, min_count, neg_exmpl = "glove.6B.50d.txt", 50, 1, 5
    pairs, word2index, index2word, vocab, tok_freq, opposites = init_train(glove_path, dim, min_count, neg_exmpl)    

    pretrained_weight = get_gloveready(glove_path, len(index2word), dim, word2index)
    pretrained_weight = torch.nn.functional.normalize(pretrained_weight)
    botk = load_topk(100, vocab, word2index, pretrained_weight)
    print("Glove loaded")
    negative_tbl = negative_sampling_tbl(vocab, tok_freq, len(index2word), index2word)
    print("filtered data size:", len(pairs))
    
    
    mdl = Word2Vec(len(index2word), dim, pretrained_weight)    
    if torch.cuda.is_available():
        mdl.cuda()
    
    criterion = nn.BCELoss()
    lr = 0.005
    
    # call cuda on model before passing it to the optimizer
    optimizer = optim.Adagrad(mdl.parameters(), lr)    
    
    labels = Variable(torch.ones(BATCH_SIZE, 1 + neg_exmpl) * -1.0)
    labels[:, 0] = labels[:, 0] * -1.0
    
    if torch.cuda.is_available():
        labels = labels.cuda()
        negative_tbl = negative_tbl.cuda()
            
    print('Training..')    
    pairs = list(pairs)
    prev_loss, prev_corr = 0, None
    simlex_lines = read_data('./SimLex-999/SimLex-999.txt', 3)
    
    for u in range(EPOCHS):
        mdl_norm = torch.norm(mdl.word_emb.weight.data, 2, 1, True)
        mdl.word_emb.weight.data.copy_(mdl.word_emb.weight.data/mdl_norm)
        
        smoke_test = get_sim(word2index['occur'], 10, mdl.word_emb.weight.data)
        for w in smoke_test:
            print index2word[w],
        print("")
        
        torch.save(mdl.state_dict(), './mdl_skipgme7.pth')
        shuffle(pairs) #shuffle works in place
        
        loss_epoch = per_trainepoch(mdl, pairs, negative_tbl, criterion, optimizer, labels, word2index, index2word, botk, neg_exmpl, opposites)
        print("---completed: {} and loss: {} ---".format(u, loss_epoch))        

        w2vmat = mdl.word_emb.weight.data.cpu()
        wnorm = torch.norm(w2vmat, 2, 1, True)
        w2vmat = w2vmat/wnorm
        cur_corr = getCorrelation(simlex_lines, w2vmat, word2index)
        print("Current correlation {}".format(cur_corr))
        if prev_corr is not None and cur_corr < prev_corr:
            print("early stopping at iteration {}".format(u))
            #break
            
        prev_corr = cur_corr
        
main(60)