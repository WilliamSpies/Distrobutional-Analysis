import re

with open('data.txt', 'r') as file:
    data = file.read()
data = re.sub('\n', ' ', data)
data = re.sub('-', '', data)

word_list = re.split("\s", data)
character_list = []
enviroment_list = []

for x in word_list:
    List = [x[i:i + 2]
            for i in range(len(x) - 1)]
    enviroment_list.extend(List)

for x in word_list:
    List = [x[i:i + 1]
            for i in range(len(x))]
    character_list.extend(List)
character_list = list(set(character_list))

print(enviroment_list)
print(character_list)

sigma_lists = {}
for b in character_list:
    list = []
    for a in enviroment_list:
        x = re.search(b, a)
        if x is not None:
            a = re.sub(b, '_', a, 1)
            list.append(a)
        else:
            continue
        sigma_lists.update({b: list})


print(sigma_lists)