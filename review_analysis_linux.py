#-*- coding:utf-8 -*-
import pandas as pd
from snownlp import SnowNLP
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models


#--------------------------------
def review_extraction():
    """
    评论提取
    文本去重
    """
    
    f=open('data/huizong.csv','rb')
    data=pd.read_csv(f,delimiter=',')
    brands=list(set(data[u'品牌']))
    
    
    for brandID in brands:
        data1=data[[u'评论']][data[u'品牌'] == brandID]
        l1=len(data1)
        outputfile = r'data/'+brandID+u'.txt'
        data1=data1.dropna()
        data1= pd.DataFrame(data1[u'评论'].unique())
        l2=len(data1)
        print(u'删除了%s的%s条重复评论！！！！！' %(brandID,(l1 - l2)))
        data1.to_csv(outputfile,index = False, header = False, encoding = 'utf-8')
        print(u'--------"%s"的评论已经保存---------' %brandID)
    
    f.close()
  




#压缩去重、去短（无意义的连续重复词语）-----------------------------

def words_cut():
    """
    压缩去词
    过滤短句
    """
    
    f=open('data/huizong.csv','rb')
    data=pd.read_csv(f,delimiter=',')
    brands=list(set(data[u'品牌']))
    for brandID in brands:
         infile = r'data/'+brandID+u'.txt'
         f=open(infile,'rb')
         data=pd.read_csv(f,delimiter='\t')
         yield data,brandID
         
        
        
def process(strs,brandID,reverse=False):
    """
    去除开头和结尾的重复词
    strs：文本字符串
    brandID：品牌ID
    reverse：开头和结尾标志位
    """
    s1=[]
    s2=[]
    s=[]
    if reverse :
        strs=strs[::-1]
    s1.append(strs[0])
    for ch in strs[1:]:
        #读入的当前字符与s1的首字符相同
        if ch==s1[0] :
            #s2中无字符则添加到s2中
            if len(s2)==0:
                s2.append(ch)
            else :
                if s1==s2:
                    s2=[]
                    s2.append(ch)
                else:
                    s=s+s1+s2
                    s1=[]
                    s2=[]
                    s1.append(ch)
        #读入的当前字符与s1的首字符不相同
        else :
            #多于两个字时更新，避免类如“滔滔不绝”这种情况被删除
            if s1==s2 and len(s1)>=2 and len(s2)>=2:
                s=s+s1
                s1=[]
                s2=[]
                s1.append(ch)
            else:
                if len(s2)==0:
                    s1.append(ch)
                else :
                    s2.append(ch)
    if s1==s2:
        s=s+s1
    else:
        s=s+s1+s2
    if reverse :
        return ''.join(s[::-1])
    else:
        return ''.join(s)     
    
        
    
        
if __name__=='__main__':        
    review_extraction()
    for data,brandID in words_cut():
        #data=process(data,brandID)
        
        data1 = data.iloc[:,0].apply(process,brandID=brandID)         
        data1 = data1.apply(process,brandID=brandID,reverse=True)
        #过滤短句
        data2=data1[data1.apply(len)>=4]
        print(u'--------"%s"的评论已经处理---------' %brandID)
        
        print(u'"%s"的情感分析正在处理中．．．．．．' %brandID)
        coms=[]
        coms=data2.apply(lambda x: SnowNLP(x).sentiments)
        data1=data2[coms>=0.5]
        data2=data2[coms<0.5]
        print(u'--------"%s"的情感分析已经处理---------' %brandID)
       
        #分词
        mycut = lambda s: ' '.join(jieba.cut(s)) #自定义简单分词函数
        data1 = data1.apply(mycut) #通过“广播”形式分词，加快速度。
        data2 = data2.apply(mycut)

        #保存分词结果
        outputfile1 = r'data/'+brandID+'_jd_pos_cut.txt'
        outputfile2 =  r'data/'+brandID+'_jd_neg_cut.txt'
        data1.to_csv(outputfile1, index = False, header = False, encoding = 'utf-8') #保存结果
        data2.to_csv(outputfile2, index = False, header = False, encoding = 'utf-8')          
        print('%s的pos和neg分词保存完成'%brandID)
        #去除停用词
        stoplist = r'data/stoplist.txt'
        stop = pd.read_csv(stoplist, encoding = 'utf-8', header = None, sep = 'tipdm')
        
        
        #sep设置分割词，由于csv默认以半角逗号为分割词，而该词恰好在停用词表中，因此会导致读取出错
        #所以解决办法是手动设置一个不存在的分割词，如tipdm。
        stop = [' ', ''] + list(stop[0]) #Pandas自动过滤了空格符，这里手动添加
        
        pos = pd.DataFrame(data1[:5000])
        neg = pd.DataFrame(data2[:5000])
        
        neg[1] = neg[0].apply(lambda s: s.split(' ')) #定义一个分割函数，然后用apply广播
        neg[2] = neg[1].apply(lambda x: [i for i in x if i.encode('utf-8') not in stop]) #逐词判断是否停用词，思路同上
        pos[1] = pos[0].apply(lambda s: s.split(' '))
        pos[2] = pos[1].apply(lambda x: [i for i in x if i.encode('utf-8') not in stop])
                
        #LDA主题分析
        ##负面主题分析
        neg_dict = corpora.Dictionary(neg[2]) #建立词典
        neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]] #建立语料库
        neg_lda = models.LdaModel(neg_corpus, num_topics = 3, id2word = neg_dict) #LDA模型训练
        for i in range(3):
            print('topic',i)
            print(neg_lda.print_topic(i)) #输出每个主题
        print('%s的负面主题分析完成'%brandID)
        ##正面主题分析
        pos_dict = corpora.Dictionary(pos[2])
        pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]
        pos_lda = models.LdaModel(pos_corpus, num_topics = 3, id2word = pos_dict)
        for i in range(3):
            print('topic',i)
            print(pos_lda.print_topic(i)) #输出每个主题
        print('%s的正面主题分析完成完成'%brandID)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        