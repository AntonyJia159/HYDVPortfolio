import gensim.models

model = gensim.models.Word2Vec.load("校群231126")


def near_model(temp_id):
    if sid2qq.__contains__(temp_id):
        near_list = model.wv.most_similar(sid2qq[temp_id], topn=55)
        print("Here are the list of 4 most relevant active people and their similarity") #Here are the list of 4 active people and their similarity
        count = 0
        i = 0
        while count < 5 and i < 50:
            if qq2freq[near_list[i][0]] >= 50:
                print(qq2school_id[near_list[i][0]], end=' ')
                print(near_list[i][1])
                count += 1
            i += 1

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
    school_id = input('Please enter the school ID you are interested in\n') #please enter the school ID you are interested in
    if school_id == 'exit':
        break
    print("They sent", end=' ') #he or she made n messages
    print(qq2freq[sid2qq[school_id]],end=' ')
    print("messages")
    near_model(school_id)

school_id = ' '

while school_id != 'exit':
    print('School ID to Similarity Query') # school ID to similarity query
    school_id = input('Input first SID：') # input first one
    temp_school_id = input('Input second SID：') # input second two
    print(model.wv.similarity(sid2qq[school_id], sid2qq[temp_school_id]))


