import gensim.models as word2vec
import numpy as np
from  gensim.models.word2vec import LineSentence
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:
    def __init__(self,k):
        """初始化KNN分类器"""
        assert  k>=1,"k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self,X_train,y_train):
        """根据训练数据集X_train和y_train 训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to the size of y_trains"
        assert self.k <= X_train.shape[0],\
            "the size of X_train must be as least k"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        """给定待预测数据集X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None,\
            "must fit before predict!"
        assert  X_predict.shape[1] == self._X_train.shape[1],\
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """给定单个待预测数据x,返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1],\
            "the feature number of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x)**2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self,X_test,y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return  accuracy_score(y_test,y_predict)
    def __repr__(self):
        return "KNN(k=%d)" % self.k

class TextRank(object):
    def __init__(self,simvalue,alpha,iter_num):
        self.word_list = [] #记录主题-词模型
        self.edge_dict = [] #记录节点的边连接字典
        self.simvalue = simvalue #满足最小相似性值，词与词之间可构成边
        self.alpha = alpha
        self.iter_num = iter_num #迭代次数

    # 读取文档，返回一个总的主题-词列表
    def readFile(self):
        with open("topic.txt",'r',encoding='utf-8') as tw:
            for line in tw:
                self.word_list.append(line.strip().split(" "))
            return  self.word_list

    def loadModel(self):
        #加载model
        self.path = "word2vec_xxx"
        self.model = word2vec.Word2Vec.load()
        print("")

        return self.model

    def calTR(self):

        # 首先根据词语之间相似性，构建每个词的相邻词,过滤掉相似性较小的词，返回边的集合
        word_list_len = len(self.word_list)
        term_num = 30 #每个主题下词的个数
        names = globals()
        for list , t in zip(self.word_list,range(word_list_len)):
            names['self.edge_dict_' + str(t)] = {}
            for index , word in enumerate(list):
                if word not in names.get('self.egde_dict_' + str(t)).keys():
                    tmp_set = set()
                    for i in range(term_num):
                        if i == index:
                            continue
                        word0 = list[i]
                        similarity = self.model.wv.similarity(word,word0)
                        if similarity > self.simvalue:
                            tmp_set.add(word0)
                    names.get('self.edge_dict_' + str(t))[word] = tmp_set
            names['self.matrix_' + str(t)]  = np.zeros([len(set(list)),len(set(list))])
            self.word_index = {}
            self.index_dict = {}
            for i , v in enumerate(set(list)):
                self.word_index[v] = i
                self.index_dict[i] = v
            for key in names.get('self.edge_dict_' + str(t)).keys():
                for w in names.get('self.edge_dict_' + str(t))[key]:
                    names.get('self.matrix_' + str(t))[self.word_index[key]][self.word_index[w]] = self.model.wv.similarity(key,w)
                    names.get('self.matrix_' + str(t))[self.word_index[key]][self.word_index[w]] = self.model.wv.similarity(w,key)
            for j in range(names.get('self.matrix_'+ str(t)).shape[1]):
                sum = 0
                for i in range(names.get('self.matrix_'+str(t)).shape[0]):
                    sum += names.get('self.matrix_'+ str(t))[i][j]
                for i in range(names.get('self.matrix_' + str(t)).shape[0]):
                    names.get('self.matrix_' + str(t))[i][j]  /= sum
                self.TR = np.ones([len(set(list)),1])
                for i in range(self.iter_num):
                    self.TR = (1- self.alpha) + self.alpha * np.dot(names.get('self.matrix_'+str(t)),self.TR)
                print("主题#%d:" %t)
                word_pr = {}
                for i in range(len(self.TR)):
                    word_pr[self.index_dict[i]] = self.TR[i][0]
                res = sorted(word_pr.items(),key=lambda x:x[1],reverse=True)
                list = []
                for n in range(len(res)):
                    list.append(res[n][0])
                print(list)

if __name__ == '__main__':
    tr = TextRank(0.45,0.85,800)
    tr.readFile()
    tr.loadModel()
    tr.calTR()

