# -*- coding=utf8 -*-
# from __future__ import print_function
import json
import re
import string
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.regression import LabeledPoint

from paint import res_visulization

# todo: finish experiment
# todo: word2vec
# todo: NaiveBayes

conf = SparkConf().setAppName('sentiment_analysis')
sc = SparkContext(conf=conf)
sc.setLogLevel('WARN')
sqlContext = SQLContext(sc)

# pattern of remove spacial chars
remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation))

stopwords = [u'rt', u're', u'i', u'me', u'my', u'myself', u'we', u'our',
             u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',
             u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',
             u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them',
             u'their', u'theirs', u'themselves', u'what', u'which', u'who',
             u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are',
             u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
             u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the',
             u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while',
             u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between',
             u'into', u'through', u'during', u'before', u'after', u'above', u'below',
             u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over',
             u'under', u'again', u'further', u'then', u'once', u'here', u'there',
             u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each',
             u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor',
             u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very',
             u's', u't', u'can', u'will', u'just', u'don', u'should', u'now']


def tokenize(text):
    tokens = []
    text = text.encode('ascii', 'ignore')
    # use '' replace url
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                  '',
                  text)
    text = remove_spl_char_regex.sub('', text)
    text = text.lower()

    for word in text.split():
        if word not in stopwords \
                and word not in string.punctuation \
                and len(word) > 1 \
                and word is not '``':
            tokens.append(word)

    return tokens


# 利用Spark提供的Word2Vec功能结合其提供的text8文件中的一部分单词进行了word2vec模型的预训练
lookup = sqlContext.read.parquet('./word2vecM_simple/data').alias('lookup')
lookup.printSchema()
lookup_hd = sc.broadcast(lookup.rdd.collectAsMap())


def doc2vec(document):
    doc_vec = np.zeros(100)
    tot_words = 0

    for word in document:
        try:
            # find features
            vec = np.array(lookup_hd.value.get(word))
            # print(vec)
            if vec != None:
                doc_vec += vec
                tot_words += 1

        except:
            continue

    vec = doc_vec / float(tot_words)
    return vec


# trainin
with open('tweets.json', 'r') as f:
    rawTrain_data = json.load(f)
    f.close()

train_data = []

for obj in rawTrain_data['results']:
    token_text = tokenize(obj['text'])
    tweet_text = doc2vec(token_text)
    # use LabeledPoint to polarity
    train_data.append(LabeledPoint(obj['polarity'], tweet_text))

train_rdd = sc.parallelize(train_data)
print(train_rdd)
print('------------------------------------------------------------')

# test
with open('hillary.json', 'r') as f:
    rawTest_data = json.load(f)
    f.close()

test_data = []

for obj in rawTest_data['results']:
    token_text = tokenize(obj['text'])
    tweet_text = doc2vec(token_text)
    test_data.append(LabeledPoint(obj['polarity'], tweet_text))

test_rdd = sc.parallelize(test_data)

# test randomForest
model = RandomForest.trainClassifier(train_rdd, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy='auto', impurity='gini',
                                     maxDepth=4, maxBins=32)

predictions = model.predict(test_rdd.map(lambda x: x.features))
labelsAndPredictions = test_rdd.map(lambda lp: lp.label).zip(predictions)

test_err = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test_rdd.count())

print('Test Error = ' + str(test_err))
print('Learned classification tree model:')
# print(model.toDebugString())
res_visulization(rawTest_data, predictions.collect())
