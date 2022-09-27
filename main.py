import re
import numpy as np
import pandas as pd

with open('data.txt', 'r') as file:
    data = file.read()
data = re.sub('\n', ' ', data)
data = re.sub('-', '', data)

word_list = re.split("\s", data)
character_list = []
env_list = []  #the enviroment list contains all enviroments that occur in the data including duplicates

for x in word_list:
    List = [x[i:i + 2]
            for i in range(len(x) - 1)]
    env_list.extend(List)

for x in word_list:
    List = [x[i:i + 1]
            for i in range(len(x))]
    character_list.extend(List)
character_list = list(set(character_list))

total_env = []  #the enviroment set is all posible enviroments given the character list

for x in character_list:
    total_env.append('_'+x)
    total_env.append(x+'_')

env_dict = {}
for b in character_list:
    list = []
    for a in env_list:
        x = re.search(b, a)
        if x is not None:
            a = re.sub(b, '_', a, 1)
            list.append(a)
        else:
            continue
        env_dict.update({b: list})

count_dict = {}
for a in env_dict:
    list = []
    for b in total_env:
        list.append(env_dict[a].count(b))
    vct = np.array(list)
    count_dict.update({a : vct})

df = pd.DataFrame(count_dict, total_env)

def normal(x):
    (x.div(len(env_list))).div((df.sum(axis=1).div(len)))


sum_vct = df.sum(axis=1)
prob_vct = sum_vct.div(len(env_list))

print(df)
