## Independent Study 
----
Experiments tried include

* **Deciding on the dataset variant of PPDB**: Trying XXL, XL gave a lot of pairs of words which were non-informative, redundant and most importantly high variance. For instance for the word "discarded" it is in pairs with 
 - 651 other words if we look into the XXL database
 - 251 in the XL
 - 115 in the L

 The final count also filters out all pairs
	 * Which have a PPDB2.0Score of less than 3.3.
	 * We compute the *edit distance* between each word of the PPDB pair to threshold it to capture word overlap and redundancy between a word pair. It is important to remove redundant pairs else the glove vectors are not updated since every word is closely in context of every other word and a discriminating signal is not provided to differentially update the word embedding

* **Deciding on the Loss function**: There are many variants of the loss function. 
	* The one used in the Weitling Paper depends on only a single negative sample and is equivalent to a max margin loss function. Further, this method choose the negative sample from the same mini-batch and is made as similar as possible to the target-context word
	* I have followed the skipgram negative sampling loss function provided in the Word2vec paper with approximately 60 negative examples of context per positive context word for the given target word
		* Under this section we have two variants,  one which uses a sigmoid output coupled with a binary cross entropy loss and the other is simply calculating the logsigmoid and negative logsigmoid and consider it as the total loss. I did not find a noticeable difference between either of these two sub variants

* **Deciding on the batch size**: Through experimenting a very counter intuitive feature I noticed that significantly impacts the optimization process is the batch size. I initially tried a batch size of value 100,000 50,000 where basically I was trying to pack in as many samples as my system RAM could support. This turns out to be very incorrect as I was noticing no decrease in the loss and correspondingly tried smaller batches of size 100, 500, 1000 and noticed 100 to work best

* **Embedding weight initializations**: The skip-gram model of word2vec utilizes two matrices the word embedding matrix which is used to lookup embeddings for the target words and the context embedding matrix which is used to lookup word context embeddings. In our model we follow the same convention but we initialize these matrices to the word embeddings of *glove*. Note: this is *different* from the model of Weitling et al who use only one matrix for both the target word and context word embedding lookup. Within this purview we could try many experiments
	* *Random initialization vs pre-trained glove*: With random initialization we get gibberish results because the data set we have of about 200,000 pairs of sentences is not supportive of full training but suitable of fine tuning which we follow. 
	* *Dimension of embedding*: As expected increasing the number of dimensions does increase the nearest neighbor quality for a given query word but for our experiments we set it to *50* since with 300 embedding the computation time is expensive

* **Different Optimizers**: I experimented with three different optimizers - (Adam, Adagrad, SGD) and found the best to be SGD with a constant learning rate of 0.1. Although, I believe that implementing a learning rate decay mechanism may improve the model performance. We notice converge in loss value after approximately 10 epochs 


### Sample results
![Results](https://ibb.co/fzY8dR "Results and vocab size")