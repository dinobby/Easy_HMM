# By tostq <tostq216@163.com>
# blog: blog.csdn.net/tostq

import numpy as np
import hmm
from hmmlearn.hmm import MultinomialHMM
state_M = 4
word_N = 0

state_list = {'B':0,'M':1,'E':2,'S':3}

# 獲得某詞的分詞結果
# 如：（我：S）、（你好：BE）、（恭喜發財：BMME）
def getList(input_str):
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append(3)
    elif len(input_str) == 2:
        outpout_str = [0,2]
    else:
        M_num = len(input_str) -2
        M_list = [1] * M_num
        outpout_str.append(0)
        outpout_str.extend(M_list)
        outpout_str.append(2)
    return outpout_str


# 預處理詞典：RenMinData.txt_utf8
def precess_data():
    ifp = file("RenMinData.txt_utf8")
    line_num = 0
    word_dic = {}
    word_ind = 0
    line_seq = []
    state_seq = []
    # 保存句子的字序列及每個字的狀態序列，並完成字典統計
    for line in ifp:
        line_num += 1
        if line_num % 10000 == 0:
            print line_num

        line = line.strip()
        if not line:continue
        line = line.decode("utf-8","ignore")

        word_list = []
        for i in range(len(line)):
            if line[i] == " ":continue
            word_list.append(line[i])
            # 建立單詞表
            if not word_dic.has_key(line[i]):
                word_dic[line[i]] = word_ind
                word_ind += 1
        line_seq.append(word_list)

        lineArr = line.split(" ")
        line_state = []
        for item in lineArr:
            line_state += getList(item)
        state_seq.append(np.array(line_state))
    ifp.close()

    lines = []
    for i in range(line_num):
        lines.append(np.array([[word_dic[x]] for x in line_seq[i]]))

    return lines,state_seq,word_dic

# 將句子轉換成字典序號序列
def word_trans(wordline, word_dic):
    word_inc = []
    line = wordline.strip()
    line = line.decode("utf-8", "ignore")
    for n in range(len(line)):
        word_inc.append([word_dic[line[n]]])

    return np.array(word_inc)

X,Z,word_dic = precess_data()
wordseg_hmm = hmm.DiscreteHMM(4,len(word_dic),5)
wordseg_hmm.train_batch(X,Z)

print("startprob_prior: ", wordseg_hmm.start_prob)
print("transmit: ", wordseg_hmm.transmat_prob)

sentence_1 = "我要回家吃飯"
sentence_2 = "中國人民從此站起來了"
sentence_3 = "經黨中央研究決定"
sentence_4 = "江主席發表重要講話"

Z_1 = wordseg_hmm.decode(word_trans(sentence_1,word_dic))
Z_2 = wordseg_hmm.decode(word_trans(sentence_2,word_dic))
Z_3 = wordseg_hmm.decode(word_trans(sentence_3,word_dic))
Z_4 = wordseg_hmm.decode(word_trans(sentence_4,word_dic))

print("我要回家吃飯: ", Z_1)
print("中國人民從此站起來了: ", Z_2)
print("經黨中央研究決定: ", Z_3)
print("江主席發表重要講話: ", Z_4)
