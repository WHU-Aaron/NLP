# encoding=utf-8
#jieba的三种模式
import jieba
str= "2018汉马全马男子冠军诞生！摩洛哥选手卫冕"

seg_list= jieba.cut(str, cut_all=True)
print("全模式: " + "/".join(seg_list)) # 全模式
print("-------------------------------------")

seg_list= jieba.cut(str)
print("默认模式: " + "/".join(seg_list)) # 默认模式= 精确模式
print("-------------------------------------")

seg_list= jieba.cut_for_search(str) # 搜索引擎模式
print("搜索引擎模式: " + "/".join(seg_list))