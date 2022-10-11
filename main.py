import re
import numpy as np

with open('data.txt', 'r', encoding='utf-8') as file:
    data = file.read()


data.lower()
data = re.sub('\n', ' ', data)
data = re.sub('\W', ' ', data)
data = re.sub('\d', ' ', data)


word_list = list(filter(None, re.split(' ', data)))

for count, str in enumerate(word_list):
    word_list[count]= f'#{str}#'

print(word_list)

character_list = []
env_list = []  # the environment list contains all environments that occur in the data including duplicates

for x in word_list:
    List = [x[i:i + 2]
            for i in range(len(x) - 1)]
    env_list.extend(List)

for x in word_list:
    List = [x[i:i + 1]
            for i in range(len(x))]
    character_list.extend(List)
character_list = list(set(character_list))
character_list.remove('#')

total_env = []  # the environment set is all possible environments given the character list
# total_env = [[f"_{x}", f"_{x}"] for x in character_list]
for x in character_list:
    total_env.append(f'_{x}')
    total_env.append(f'{x}_')

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

count_list = []
for a in env_dict:
    list = []
    for b in total_env:
        list.append(env_dict[a].count(b))
    count_list.append(list)
count_matrix = np.array(count_list)

normal_matrix = ((count_matrix / len(env_list))
                 / (np.matmul(np.expand_dims([sum([row[i] for row in count_matrix / len(env_list)])
                                              for i in range(0,len((count_matrix / len(env_list))[0]))], axis=1),
                              np.expand_dims([sum(row) for row in count_matrix / len(env_list)], axis=0)))
                 .T)

np.set_printoptions(edgeitems=len(env_list))
print(normal_matrix)
