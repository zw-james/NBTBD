#coding=utf-8
import  sys
sys.path.append("../")



import jieba
import jieba.analyse
import jieba.posseg
import random
import os
import math



def ConverData():
    i = 0
    tag = 0
    WordList = []
    WordIDDic = {}
    KeyList = []
    KeyDic = {}
    TrainingPercent = 0.9

    fo = open("fdata_all.txt", "r")
    trainOutFile = open("tbd.train.all", "w")
    testOutFile = open("tbd.test.all", "w")

    for line in fo.readlines():
        i = i+1
        line = line.strip()
        num_n = line.find('\t')

        if (num_n == -1):
            continue
        key_all = line[0:num_n]
        key_F = line[0:line.find('_')]
        key_S = line[line.find('_')+1:num_n]
        key_F = key_F.replace('\xef\xbb\xbf','')
        key_S = key_S.replace('\xef\xbb\xbf', '')

        vale = line[line.index('\t'):].replace('\t', ' ').strip()
        words = jieba.cut(vale, cut_all=False)

        if(key_F+'train' not in KeyDic):
            filename_train = 'tbd.train.' + key_F;
            filename_test = 'tbd.test.' + key_F;
            f_train = open(filename_train,"w");
            KeyDic[key_F + 'train'] = f_train;
            f_test = open(filename_test, 'w');
            KeyDic[key_F + 'test'] = f_test;
        else:
            f_train =KeyDic[key_F+'train']
            f_test =KeyDic[key_F + 'test']

        # 计算随机数
        rd = random.random()
        outfile = testOutFile
        outfile_s = f_test
        if rd < TrainingPercent:
            outfile = trainOutFile
            outfile_s = f_train


        outfile.write(key_F + " ")
        outfile_s.write(key_S + " ")

        for word in words:
            if(len(word.strip()) < 1):
                continue
            if word not in WordIDDic:
                WordList.append(word)
                WordIDDic[word]=len(WordList)
            outfile.write(str(WordIDDic[word])+" ")
            outfile_s.write(str(WordIDDic[word]) + " ")


        outfile.write("#"+str(i)+"\n")
        outfile_s.write("#"+str(i)+"\n")

    trainOutFile.close()
    testOutFile.close()
    fo.close()

""" 
def LoadData(TrainFileName):
    i = 0
    infile = file(TrainFileName, 'r')
    sline = infile.readline().strip()
    while len(sline) > 0:
        pos = sline.find("#")
        if pos > 0:
            sline = sline[:pos].strip()
        words = sline.split(' ')
        if len(words) < 1:
            print("Format error!")
            break
        classid = int(words[0])
        if classid not in ClassFeaDic:
            ClassFeaDic[classid] = {}
            ClassFeaProb[classid] = {}
            ClassFreq[classid] = 0
        ClassFreq[classid] += 1
        words = words[1:]
        # remove duplicate words, binary distribution
        # words = Dedup(words)
        for word in words:
            if len(word) < 1:
                continue
            wid = int(word)
            if wid not in WordDic:
                WordDic[wid] = 1
            if wid not in ClassFeaDic[classid]:
                ClassFeaDic[classid][wid] = 1
            else:
                ClassFeaDic[classid][wid] += 1
        i += 1
        sline = infile.readline().strip()
    infile.close()
    print i, "instances loaded!"
    print len(ClassFreq), "classes!", len(WordDic), "words!"
"""

def ComputeModel(TrainFileName):
    DefaultFreq = 0.1
    i = 0;
    #先验概率
    classP = {}
    #每个类别产生该对象的概率
    classWord ={}
    #单词默认概率
    classWordDefault ={}
    #单词表
    wordlist={}
    #用于方便统计各类型中单词总量
    classWordNum = {}

    fo = open(TrainFileName,'r');
    for line in fo.readlines():
        i=i+1;
        line = line.strip();
        nPos=line.find('#');
        lineNum=line[nPos+1:];
        line=line[0:nPos].strip();
        words=line.split(' ')
        classid = int(words[0])
        if(classid not in classP):
            classP[classid] = 0
            classWordNum[classid] = 0
            classWord[classid]={}
        classP[classid] += 1
        #去掉标示符号
        words = words[1:]
        for word in words:
            #统计词个数
            if word not in wordlist:
                wordlist[word] = 1
            #统计每个类型中的单词的数量
            if(word not in classWord[classid]):
                classWord[classid][word] = 1
            else:
                classWord[classid][word] += 1
            #记录每个类型单词的总数
            classWordNum[classid] +=1
    fo.close();

    #i=float(len(wordlist)*DefaultFreq +i)

    #计算每个种类所占比例
    for classid in classP.keys():
        classP[classid] = float(classP[classid])/float(i);

    #计算每个种类每个词在该种类中所占比例   添加DefaultFreq进行 拉普拉斯平滑修正
    for classid in  classWord.keys():
        nsum=(float)(classWordNum[classid] + len(wordlist)* DefaultFreq)
        for word in classWord[classid].keys():
            classWord[classid][word] = float(classWord[classid][word] + DefaultFreq) / nsum
        classWordDefault[classid]=float(DefaultFreq)/nsum

    return classP,classWord,classWordDefault,wordlist

def Predict(TestFileName,TrainFileName):
    classP, classWord, classWordDefault, wordlist = ComputeModel(TrainFileName)

    fo = open(TestFileName,'r')
    TrueLabelList = []
    PredLabelList = []
    sortDic = {}

    for line in fo.readlines():
        nPos = line.find('#')
        line = line[0:nPos].strip();
        words = line.split(' ');
        TrueLabelList.append(int(words[0]))
        words = words[1:]
        for classid in classP.keys():
            sortDic[classid]=math.log(classP[classid])

            for word in words:
                if word not in wordlist:
                    continue;
                if word in classWord[classid]:
                    sortDic[classid] +=math.log(classWord[classid][word])
                else:
                    sortDic[classid] +=math.log(classWordDefault[classid])

        maxProb = max(sortDic.values())
        for classid in sortDic.keys():
            if   sortDic[classid] == maxProb :
                PredLabelList.append(classid)
                break;
    fo.close();

    return  TrueLabelList,PredLabelList


ConverData();

#classP,classWord = ComputeModel('tbd.train.all')

TList,PList = Predict('tbd.test3','tbd.train3')
i = 0
outfile = file('out.ttt', 'w')
while i < len(TList):
    outfile.write(str(TList[i]))
    outfile.write(' ')
    outfile.write(str(PList[i]))
    outfile.write('\n')
    i += 1

accuracy = 0
i = 0
while i < len(TList):
    if TList[i] == PList[i]:
        accuracy += 1
    i += 1
accuracy = (float)(accuracy)/(float)(len(TList))
print "Accuracy:",accuracy

outfile.close()
