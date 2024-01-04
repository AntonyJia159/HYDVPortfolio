
import gensim.models
import random


from numpy import log

model = gensim.models.Word2Vec.load("校群231126")


def near_model(temp_id, freq_bar=100, topn=50, balance=1):
    if sid2qq.__contains__(temp_id):
        near_list = model.wv.most_similar(sid2qq[temp_id], topn=topn)
        ids = []
        weights = []
        for a in near_list:
            if qq2freq[a[0]] >= freq_bar:
                ids.append(a[0])
                weights.append(min((a[1] * log(qq2freq[a[0]]) + balance), 5.5+balance))
        k = int(0.75 + len(ids)/3)
        # print(weights)
        print("以下是抽取的", k, "个相关人物！") # Here are top-k relevant people
        for b in random.choices(ids, weights, k=k):
            print(qq2school_id[b], end=' ')
        print()



fo = open(r'./in/231126_qq2school_id.txt')
qq2school_id = eval(fo.read())
fo.close()

fo = open(r'./in/231126_qq2freq.txt')
qq2freq = eval(fo.read())
fo.close()

sid2qq = {}
for qq,sid in qq2school_id.items():
    if sid is not None:
        sid2qq[sid] = qq

school_id = ' '

while school_id != 'exit':
    school_id = input('请输入你感兴趣的学号') #please enter the school ID you are interested in
    print("ta发言了",qq2freq[sid2qq[school_id]],"次")
    near_model(school_id)




