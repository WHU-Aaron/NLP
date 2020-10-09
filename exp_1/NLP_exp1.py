import pickle
import gensim
from gensim.models import Word2Vec, word2vec
from gensim.corpora.dictionary import Dictionary
#import sklearn
import matplotlib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

#工作目录
addr = r'C:\Users\loeoe\Desktop\NLP1'


#==========================1.读取预训练语料=======================================
print("选择分词后的文本作为训练语料...")
#这里输入数据格式为txt,一行为一个样本
with open("outputs2.txt", "r", encoding="utf-8") as f:
    raw_sentences = f.readlines()
    sentences = []
    for i in raw_sentences:
        sentences.append(i.split(' '))
print(sentences[0])
#=====================2.训练Word2vec模型（可修改参数）...====================
print('训练Word2vec模型...')
model = Word2Vec(sentences,
                 size=100,  # 词向量维度
                 min_count=1,  # 词频阈值
                 window=5,
                 workers=4)
           

#根据已经保存的模型加载模型，不必重复训练
#model = gensim.models.Word2Vec.load(r'C:\Users\loeoe\Desktop\NLP1w2v_100.model')


#保存模型
print(u"保存w2v模型...")
model.save(addr + 'w2v_100.model')  # 保存模型
print("保存w2v模型的位置： ", addr + 'w2v_100.model', '\n')


#计算相似度
print(model.wv.most_similar(positive=['武汉'], topn=5))
print(model.wv.most_similar(positive=['湖北', '成都'], negative=['武汉'], topn=5))


The_list=['江苏', '南京', '成都', '四川', '湖北', '武汉','河南', '郑州',
'甘肃', '兰州', '湖南', '长沙', '陕西', '西安', '吉林', '长春', '广东',
'广州', '浙江', '杭州'] 

vectors = [] # positions in vector space
labels = [] # keep track of words to label our data again later
for word in The_list:
    vectors.append(model.wv[word])
    labels.append(word)

# convert both lists into numpy vectors for reduction
# vectors = np.asarray(vectors)
# labels = np.asarray(labels)

#图形中的中文显示
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


#降到维度2
pca = PCA(n_components=2)
results = pca.fit_transform(vectors)

print(results)

#可视化
#sns.scatterplot(x=results[:, 0], y=results[:, 1])
plt.figure(figsize=(14,10))
plt.scatter(results[:,0], results[:,1])
for i in range(20):
    x = results[i][0]
    y = results[i][1]
    plt.text(x, y, labels[i])
plt.show()

if __name__ == "__main__":
    pass