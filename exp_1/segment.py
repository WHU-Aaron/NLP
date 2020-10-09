# encoding=utf-8
import jieba
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


#C:\Users\loeoe\Desktop\exp1_corpus.txt
#以test.txt作为小的样本进行测试
path=r'C:\Users\loeoe\Desktop\test.txt'

f=open(path,'r',encoding='utf-8')
'''
for str in f:
    #按默认模式（精确模式）进行分词处理
    str_list=jieba.cut(str) 
    print('/'.join(str_list))

path = get_tmpfile("word2vec.model")
model = Word2Vec(path, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
print(model)
'''
# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    #sentence_depart = jieba.cut(sentence.strip(),HMM=False)
    sentence_depart=jieba.cut(sentence)
    
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        #outstr += word+'/'
        outstr=outstr+word+'/'
    return outstr


outputs=open(r'C:\Users\loeoe\Desktop\NLP1\testoutput.txt','w',encoding='utf-8')

# 将输出结果写入ou.txt中
for line in f:
    line_seg = seg_depart(line)
    outputs.write(line_seg)

outputs.close()
f.close()




