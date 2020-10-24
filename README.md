# NLP
#### exp_1
1. 使用 jieba 分词工具进行分词，使用方法： jieba.cut(text) ；
2. 使用 gensim 中的 Word2Vec 模型训练词向量： model = Word2Vec(common_texts, size=100,
window=5, min_count=1, workers=4) ；
3. 使用训练好的词向量对指定的词（2个例子）进行相关性比较： model.similarity('中国','中
华') ；
4. 使用训练好的词向量选出与指定词（2个例子）最相似的5个词：
model.wv.most_similar(positive=['武汉'], topn=5) ；
5. 使用训练好的词向量选出与指定词类比最相似的5个词（2个例子），如湖北 - 武汉 + 成都 = 四
川： model.wv.most_similar(positive=['湖北', '成都'], negative=['武汉'], topn=5) ；
6. 使用 sklearn 中的 PCA 方法对列表 ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河
南', '郑州', '甘肃', '兰州', '湖南', '长沙', '陕西', '西安', '吉林', '长春', '广东',
'广州', '浙江', '杭州'] （可换成其他）中的所有词的词向量进行降维并使用 seaborn 和
matplotlib 将其可视化：
#### exp_2
TextCNN 模型,BiLSTM 模型
经过调参，最好的参数组合
TextCNN:    87.1%
BiLSTM:     87.9%
