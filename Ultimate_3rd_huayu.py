# coding=UTF-8

import re
import gensim.models
import numpy as np
from gensim.models import Word2Vec
from matplotlib import pyplot as plt, cm
from sklearn.manifold import TSNE


viridis = cm.get_cmap('plasma', 18)
batchs = []
new_batch = []
times = []
qq2code = {}
code2qq = {}
qq2school_id = {}
qq2freq = {}
qqs = []
tempqqs = []
ids = []


def extract_school_id(line):
    school_id = re.search(r'(?<=[\s】])(?:1[0-9]|2[0-5]|0[89])[1-8][015678]\d', line)
    if school_id is not None:
        school_id = school_id.group()
    return school_id

def sort_by_key(d):
    return sorted(d. items(), key=lambda k: k[0])


def process_school_id(temp_id, t_temp_id, seq):  # 递归的为新QQ号寻找代号，小号会被分配为22859a、22859b，以此类推
    if temp_id is None:
        return None
    while ids.__contains__(temp_id):
        return process_school_id(t_temp_id+chr(seq), t_temp_id, seq+1)
    return temp_id


def output():  # 输出到文件，主要是hy、qqs、qq2school_id, 方便后续利用
    out = './out/231126_hy.txt'
    f = open(out, 'w')
    for q in qqs:
        f.write(q)
        f.write(' ')
        f.write(str(qq2school_id.get(q, 'Unexpected')))
        f.write(' ')
        f.write(str(qq2freq[q]))
        f.write('\n')
    f.close()
    out = './in/231126_qq2school_id.txt'
    f = open(out, 'w')
    f.write(repr(qq2school_id))
    f.close()
    out = './in/231126_qqs.txt'
    f = open(out, 'w')
    f.write(repr(qqs))
    f.close()
    output_freq()

def output_freq():
    out = './in/231126_qq2freq.txt'
    f = open(out, 'w')
    f.write(repr(qq2freq))
    f.close()


def get_embeddings(model):  # 将词与嵌入的对应转化为字典格式
    embeddings = {}
    for word in qqs:
        embeddings[word] = model.wv[word]

    return embeddings



def transform(embeddings):  # t-SNE降维模块，不用动，直接调用即可
    emb_list = []

    for k in qqs:
        if qq2school_id[k] is None:
            jie = 0.21
        else:
            jie = float(qq2school_id[k][:2])/100
        emb_list.append(np.append(embeddings[k], jie))

    emb_list = np.array(emb_list)
    down = TSNE(n_components=2, learning_rate=300, init='pca', perplexity=25, metric='cosine')
    node_pos = down.fit_transform(emb_list)

    return node_pos


def plot_embeddings(embeddings):  # 使用Matplotlib画图，可不使用

    node_pos = transform(embeddings)

    color_idx = {}

    for i in range(len(qqs)):

        temp_school_id = qq2school_id[qqs[i]]

        if temp_school_id is None:
            temp_school_id = '0'

        color_idx.setdefault(temp_school_id, [])

        color_idx[temp_school_id].append(i)

    color_idx = dict(sort_by_key(color_idx))

    print(color_idx)

    for c, idx in color_idx.items():
        if not c == '0':

            generation = c[0:2]
            t = viridis((int(generation) - 7) / 19)
        else:
            generation = 'unknown'
            t = viridis(0)

        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], s=75, alpha=0.6, c=t, label=generation)  # c=node_colors)
        #plt.annotate(c, xy=(node_pos[idx, 0], node_pos[idx, 1]))


    plt.legend()

    plt.show()


def train(sentences, embed_size=32, window_size=5, workers=3, epochs=5, **kwargs):
    # 训练word2vec的函数，这里用关键词表传入参数
    kwargs["sentences"] = sentences
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = embed_size
    kwargs["sg"] = 1
    kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
    kwargs["workers"] = workers
    kwargs["window"] = window_size
    kwargs["epochs"] = epochs

    print("Learning embedding vectors...")
    model = Word2Vec(**kwargs)
    print("Learning embedding vectors done!")

    return model


def extract_log_titles(file_path):  # 提取每一条的记录头（即包含日期、时间、昵称、QQ号的那一部分）
    re_pat = r'20[\d-]{8}\s[\d:]{7,8}\s+[^\n]+(?:\d{5,11}|@\w+\.[comnetcn]{2,3})[)>]'  # 这个规律是“20”加8个或数字或-的字符，
    # 一个空格，7、8个数字或：字符，一个空格，非换行符号若干，一个分叉式：匹配QQ号格式或者邮箱格式

    with open(file_path, 'r', encoding='utf-8', errors=' ignore') as reader:
        txt = reader.read()

    log_content_arr = re.split(re_pat, txt)[1:]

    return re.findall(re_pat, txt)  # 返回所有记录头的列表

# 该版本程序适用于路人甲记录


def pre_analysis():  # 分析一整个文档并准备语料
    temps = []  # 这个列表是用来准备伪句子格式的，等每一句准备完，它会被加入到总语料，并重新初始化
    code = 0  # 这个变量主要用于记录当前是否是某个伪句子里面的第一个词
    fn = r'./in/log.txt'  # 路径
    datetemp = ''
    for head in extract_log_titles(fn):  # 遍历记录头组件
        time = re.search(r'20[\d-]{8}\s[\d:]{7,8}', head).group()  # 匹配日期和时间
        #print(time,end=' ')
        date = re.search(r'20[\d-]{8}', time).group()  # 从日期和时间中，匹配日期
        if datetemp != date:
            print(date)
            datetemp = date
        qq = re.search(r'(?:(?<=[(])\d+|(?<=[<])[^>]+)', head).group()  # 匹配QQ号或者邮箱
        times.append(date)  # 在时间总表中添加当前日期
        temp_temp_id = extract_school_id(head)  # 这里写的这么复杂是为了规避小号问题。temp_temp_id是提取可能存在的原始学号，
        # 也有可能没有，即返回None。
        school_id = process_school_id(temp_temp_id, temp_temp_id, 97)
        # 这一句是规避小号，第一项是要检索的id，第二项是id的根，用于生成小号代号，97是ascii“a”，用作标记。
        # 这个函数本身会递归的检索学号数据，如果某个学号已经存在但匹配另一个qq号，比如22859，函数就会进一步查询22859a，
        # 再查询22859b，直到查询到一个该学号的未占用版本为止
        if not qq2code.__contains__(qq):  # 这一句是为了确认当前发言者是不是第一次在记录中出现
            qq2freq.setdefault(qq, 1)  # 初始化该qq号发言计数为1
            qq2code[qq] = code
            code2qq[code] = qq
            # qq2code 其实属于早期的功能，现版本可以视为废代码。但code本身是有意义的，不能删
            qqs.append(qq)
            # 在qqs表中加入新qq号。这个表会用来储存所有QQ号，遍历嵌入，同时输出到文件下次直接用
            qq2school_id[qq] = school_id
            # 建立qq号与正规化的代号的对应
            ids.append(school_id)
            # 更新学号列表
            temps.append(qq)
            # 在伪句子语料中添加当前qq
        elif qq2school_id[qq] is None:  # 在此前未检测到合法学号的的情况下，会一直试图判定并更新学号
            qq2school_id[qq] = school_id
            # 同上
            ids.append(school_id)
            temps.append(qq)
        else:
            qq2freq[qq]+=1  # 这是最普遍的情况，非新非空学号
            if len(temps) > 0:  # 检测当前伪句子是否空。这一段处理是为了减少“长叠词”现象，即如果一个人连续发了多条信息，会合并为一个“信息”
                #  记录，这样的能更好捕捉群友的交互特征
                if temps[-1] != qq:  # 当前伪句子已有东西，检测这一条消息和上一条是不是同一个人说的
                    temps.append(qq)  # 不是才加入伪句子
            else:
                temps.append(qq)  # 如果是空的就加进去

        if code > 0 and not times[-1] == times[-2]:  # 伪句子终结判断。目前是以一天的发言作为一个伪句子，当跨过0点（可惜是路人甲时间）
            # 会新建一个伪句子
            batchs.append(temps)  # 在总语料中加入当前伪句子
            temps = []  # 重新初始化伪句子
        code += 1  # code增加



pre_analysis()  #  运行独缺

#fo = open(r'./in/231126_qq2school_id.txt')
#qq2school_id = eval(fo.read())
#fo.close()

#fo = open(r'./in/231126_qqs.txt')
#qqs = eval(fo.read())
#fo.close()

# 上面这两段是适用于已经处理好数据和训练好模型，想第二次运行该程序。此时可以直接从in目录下读取之前格式化输出好的qqs和qq2school_id
# 这样要把pre_analysis注释掉

if input("train?y/n") == 'y':
    model = train(batchs)
    model.save("校群231126")
else:
    model = gensim.models.Word2Vec.load("校群231126")

#plot_embeddings(model.wv)
#print(transform(model.wv))

for pos in transform(model.wv):
    print(pos[0], pos[1])

output()



