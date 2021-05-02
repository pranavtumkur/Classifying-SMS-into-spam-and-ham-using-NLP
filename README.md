# Classifying-SMS-into-spam-and-ham-using-NLP

In this project, we do a text analysis using NLP to classify whether a text SMS is spam or ham (slang for a normal SMS which is not-spam). We'll do this by training a machine learning model to learn to discriminate between ham/spam automatically. Then, with the trained model, we'll be able to classify arbitrary unlabeled messages as ham or spam. We'll follow the below steps:

![1_nBgCTU_hAVG00eYkcRf6Mw](https://user-images.githubusercontent.com/65482013/85939311-d6699080-b931-11ea-908c-ed07244f1706.jpg)

### 1. Data Cleaning and formatting

### 2. EDA

### 3. Text pre-processing

Converting the corpus into lists of tokens (also known as [lemmas](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)) using CountVectorizer method in the feature_extraction library. This model will convert a collection of text documents to a matrix of token counts.

We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message.

For example:


<table border = “1“>
<tr>
<th></th> <th>Message 1</th> <th>Message 2</th> <th>...</th> <th>Message N</th> 
</tr>
<tr>
<td><b>Word 1 Count</b></td><td>0</td><td>1</td><td>...</td><td>0</td>
</tr>
<tr>
<td><b>Word 2 Count</b></td><td>0</td><td>0</td><td>...</td><td>0</td>
</tr>
<tr>
<td><b>...</b></td> <td>1</td><td>2</td><td>...</td><td>0</td>
</tr>
<tr>
<td><b>Word N Count</b></td> <td>0</td><td>1</td><td>...</td><td>1</td>
</tr>
</table>

Since there are so many messages, we can expect a lot of zero counts for the presence of that word in that document. Therefore to save on memory and processing speed, we use a **Sparse Matrix** to store our 'Bag of Words' model. For this model, the sparsity is 0.079, i.e **7.9%** of the values in the matrix are non-zero.

### 4. Vectorization

We'll convert the lemmas above, into vectors that machine learning models can understand. We'll do that in three steps using the [bag-of-words](http://en.wikipedia.org/wiki/Bag-of-words_model) model. We'll do that in three steps using the bag-of-words model:

- Count how many times does a word occur in each message (Known as term frequency)
- Weigh the counts, so that frequent tokens get lower weight (inverse document frequency) using TF-IDF (*term frequency-inverse document frequency*)
- Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

### 5. Training a model

With the pre-processing and the vectorization, we can actually use almost any sort of classification algorithms. For a [variety of reasons](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf), the Naive Bayes classifier algorithm is a good choice.

**Evaluating this trained model comparing the training datasets, actual values, we get an accuracy of 96% which is a good enough accuracy for text SMS classification!**
