#### 实验内容介绍

* ##### jieba

  ###### 定位：
  Chinese text segmentation: built to be the best Python Chinese word segmentation module.

  ###### 特点：

  + 支持四种分词模式：
    + 精确模式，试图将句子最精确地切开，适合文本分析；
    + 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
    + 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
    + paddle模式，利用PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词。
  + 支持繁体分词
  + 支持自定义词典
  + MIT 授权协议

  ###### 算法介绍：
  
  + 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图(DAG)
  + 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
  + 对于未登录词，采用了基于汉字成词能力的HMM 模型，使用了Viterbi 算法
  
  ###### 功能介绍
  
  * 分词
  * 添加自定义词典
  * 关键词提取
  * 词性标注
  * 并行分词

* ##### gensim

  ###### 定位：
  

是一个Python库，用于使用大型语料库进行主题建模，文档索引和相似性检索。目标受众是自然语言处理（NLP）和信息检索（IR）社区。

###### 作用：

  + All algorithms are **memory-independent** w.r.t. the corpus size (can process input larger than RAM, streamed, out-of-core),
  + Intuitive interfaces
    + easy to plug in your own input corpus/datastream (trivial streaming API)
    + easy to extend with other Vector Space algorithms (trivial transformation API)
  + Efficient multicore implementations of popular algorithms, such as online **Latent Semantic Analysis (LSA/LSI/SVD)**, **Latent Dirichlet Allocation (LDA)**, **Random Projections (RP)**, **Hierarchical Dirichlet Process (HDP)** or **word2vec deep learning**.
  + **Distributed computing**: can run *Latent Semantic Analysis* and *Latent Dirichlet Allocation* on a cluster of computers.
  + Extensive documentation and Jupyter Notebook tutorials.

  

#### 实验环境

* python 3.7
* jieba
* gensim 
* sklearn
* matplotlib 
* numpy
* seaborn



#### 实验过程

##### 1.使用 jieba 分词工具进行分词

​	首先安装所需要的python包

![image-20200929094817679](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20200929094817679.png)



![image-20200929094847620](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20200929094847620.png)

导入文件，利用jieba自带的cut方法进行分词

```python
import jieba
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
path=r'C:\Users\loeoe\Desktop\exp1_corpus.txt'

f=open(path,'r',encoding='utf-8')
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    #sentence_depart = jieba.cut(sentence.strip(),HMM=False)
    sentence_depart=jieba.cut(sentence)
    
    # 输出结果为outstr
    outstr = ''
    for word in sentence_depart:
        outstr += word+'/'
    return outstr

'''
for str in f:
    #按默认模式（精确模式）进行分词处理
    str_list=jieba.cut(str) 
    print('/'.join(str_list))
'''

outputs=open(r'C:\Users\loeoe\Desktop\Python Codes\output.txt','w',encoding='utf-8')

# 将输出结果写入ouuputs.txt中
for line in f:
    line_seg = seg_depart(line)
    outputs.write(line_seg + '\n')

outputs.close()
f.close()
```

去除停用词：

```python
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
```

结果如下：

![image-20201006190819673](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006190819673.png)

##### 2. 使用 gensim 中的 Word2Vec 模型训练词向量

##### 关于Word2vec：

* Word2vec，是一群用来产生词向量的相关模型。这些模型为浅层双层的神经网络，用来训练以重新建构语言学之词文本。网络以词表现，并且需猜测相邻位置的输入词，在word2vec中词袋模型假设下，词的顺序是不重要的。

* 训练完成之后，word2vec模型可以把每个词映射到一个向量，来表示词与词之间的关系。该向量为神经网络的隐藏层。

* Word2vec的两种模式：

  * CBOW：通过上下文来预测当前值

  * Skip-gram:通过当前词来预测上下文

* Word2vec的缺点

  * 由于词和向量是一对一的关系，所以多义词的问题无法解决。

  * Word2vec 是一种静态的方式，虽然通用性强，但是无法针对特定任务做动态优化

参数设置如下：

```python
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
```

```python
with open("outputs2.txt", "r", encoding="utf-8") as f:
    raw_sentences = f.readlines()
    sentences = []
    for i in raw_sentences:
        sentences.append(i.split(' '))
print(sentences[0])
#=====================训练Word2vec模型====================
print('训练Word2vec模型...')
model = Word2Vec(sentences,
                 size=100,  # 词向量维度
                 min_count=1,  # 词频阈值
                 window=5,
                 workers=4)
```

##### 训练中存在的问题

<img src="C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006202142137.png" alt="image-20201006202142137" style="zoom: 67%;" />

在训练时报错，经上网查询，发现进行测试训练的数据集太小，以至于词汇频率低

解决方法：

* 选取更大的数据集
* 减小参数min_count

##### 3. 用训练好的模型进行相关性比较

```python
print(model.similarity('中国','中华'))
```

<img src="C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006222255088.png" alt="image-20201006222255088" style="zoom:67%;" />

经比较，“中国”和“中华”的相关性为：0.56560117

<img src="C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006223715859.png" alt="image-20201006223715859" style="zoom:67%;" />

经比较，“毛泽东”和“毛主席”的相关性为：0.866793

<img src="C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006230650078.png" alt="image-20201006230650078" style="zoom:67%;" />

经比较，“改革”和“开放”的相关性为：0.47403577

##### 4.与指定词最相似的词

```python
print(model.wv.most_similar(positive=['武汉'], topn=5))
```

![image-20201006230227774](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006230227774.png)

和经济最相关的词为：
![image-20201006231744172](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006231744172.png)

##### 5.与指定词类比最相似的词

```python
print(model.wv.most_similar(positive=['湖北', '成都'], negative=['武汉'], topn=5))
```

![image-20201006230338751](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201006230338751.png)

##### 6. 使用 sklearn 中的 PCA 方法对列表降维和可视化

##### t-SNE

t-SNE 算法的应用需要基于这样的假设：尽管现实世界中看到的数据都是分布在高维空间中的，但是都具有很低的内在维度。也就是说高维数据经过降维后，在低维状态下更能显示出其本质特性。

##### PCA

*class* `sklearn.decomposition.PCA`(*n_components=None*, ***, *copy=True*, *whiten=False*, *svd_solver='auto'*, *tol=0.0*, *iterated_power='auto'*, *random_state=None*)

参数介绍：

* 1）**n_components**：这个参数可以帮我们指定希望PCA降维后的特征维度数目。最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。当然，我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。
* 2）**whiten** ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。
* 3）**svd_solver**：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 full则是传统意义上的SVD，使用了scipy库对应的实现。arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。



存在的问题：训练模型需要花费一定的时间，如果数据集很大的话，可以将训练好的模型保存下来，以后就不必重复训练了。

具体方法：构建有限词汇表，转存模型中的词向量为csv或其他格式，使用时载入为字典实现快速读取。

`You’ll notice that training non-trivial models can take time. Once you’ve trained your model and it works as expected, you can save it to disk. That way, you don’t have to spend time training it all over again later.`

```python
#转化为矩阵
vectors = np.asarray(vectors)
labels = np.asarray(labels)
#需要降到的维度，此处降到二维
num_dimensions = 2  # final num dimensions (2D, 3D, etc)
#降维处理
tsne = TSNE(n_components=num_dimensions, random_state=0)
vectors = tsne.fit_transform(vectors)
#查看降维之后的数据
print(vectors)
```

降维之后的向量如下：

<img src="C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201007200331976.png" alt="image-20201007200331976" style="zoom: 67%;" />

存在的问题：图像中的中文字体不能正常显示,导入mpl包即可。

```python
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
```

可视化之后如下：

```python
plt.figure(figsize=(14,10))
plt.scatter(results[:,0], results[:,1])
for i in range(20):
    x = results[i][0]
    y = results[i][1]
    plt.text(x, y, labels[i])
plt.show()
```

![image-20201007200703011](C:\Users\loeoe\AppData\Roaming\Typora\typora-user-images\image-20201007200703011.png)

#### 实验总结

1. 深入地理解NLP中的各类算法和模型（Word2Vec）。 
2. 更熟练地使用编程工具，提高了编程能力。 
3. 通过本次实验，对一些概念性的东西进行了实践。掌握了数据的预处理部分， 特别是对向量进行降维，以及对数据进行可视化的过程。
4. 实验期间，阅读了"Distributed Representations of Words and Phrases and their Compositionality"，“Efficient Estimation of Word Representations in Vector Space”等论文，提高了阅读英语论文的能力，认识了很多NLP专业名词，对以后的学习有很大帮助。 
5. NLP可以运用在很多领域 ，文本审核、广告过滤、情感分析、舆情监测、 新闻分类等，此次实验对以后在具体应用方面有着重要的先导作用。 
6. 此次实验也反映出了很多问题，对一些新算法和新工具的运用，对编程语言的掌握程度，对NLP模型的理解方面，还有很多问题。希望通过以后的不断学习和努力，学习更多更好的算法和模型，并能够运用到实践当中。 