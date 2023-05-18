import pandas as pd
import re

data_src = 'D:/data/brain/brain_hemo_214.xlsx'
term_src = '/term.xlsx'

df = pd.read_excel(data_src, skiprows=5, usecols=['readout'])
# df['readout'] = df['readout'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
df = df.dropna()
df.reset_index(drop=False, inplace=True)
# print(df['readout'][0])

wordlist = []
wordoutlist = ['ct']

df2 = pd.read_excel(term_src, usecols=['hemo'])
df2['hemo'] = df2['hemo'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
df2 = df2.dropna()
df2.reset_index(drop=False, inplace=True)

df3 = pd.read_excel(term_src, usecols=['Key Brain Terms Glossary'])
df3['Key Brain Terms Glossary'] = df3['Key Brain Terms Glossary'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
df3 = df3.dropna()
df3.reset_index(drop=False, inplace=True)

df4 = pd.read_excel(term_src, usecols=['paterehab'])
df4['paterehab'] = df4['paterehab'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
df4 = df4.dropna()
df4.reset_index(drop=False, inplace=True)

for i in range(len(df2)):
    x = df2['hemo'][i].split()
    for j in range(len(x)):
        if x[j] not in wordlist:
            wordlist.append(x[j].lower())

# for i in range(len(df3)):
#     x = df3['Key Brain Terms Glossary'][i].split()
#     for j in range(len(x)):
#         if x[j] not in wordlist:
#             wordlist.append(x[j])

for i in range(len(df4)):
    x = df4['paterehab'][i].split()
    for j in range(len(x)):
        if x[j] not in wordlist:
            wordlist.append(x[j].lower())

# print(wordlist)


strlist = []

def check_date_format(input_date):
	regex = r'\d{4}-\d{2}-\d{2}'
	return  bool(re.findall(regex, input_date))

def check_num_front(input):
    regex=r'\d'
    return re.match(regex, input)

for i in range(len(df)):
    x = df['readout'][i].splitlines()
    tmplist = []
    for j in range(len(x)):
        num = 0
        if check_date_format(x[j]):
            continue
        no_special_str = re.sub(r'[^\w]', ' ', x[j])
        no_special_str = no_special_str.strip()
        check = 0
        for k in range(len(wordoutlist)):
            if wordoutlist[k] in no_special_str.lower():
                check = 1
        for k in range(len(wordlist)):
            if wordlist[k] in no_special_str.lower():
                num = num + 1
        # if no_special_str.isnumeric() :
        #     print(no_special_str)
        #     no_special_str = no_special_str[1:]
        if num > 0 and check == 0:
            num_in_str=check_num_front(no_special_str)
            if num_in_str:
                no_special_str.rstrip(num_in_str.group())
            tmplist.append(no_special_str)


    strlist.append(tmplist)

# print(strlist)

for i in range(len(strlist)):
    newstr = ""
    for j in range(len(strlist[i])):
        newstr = newstr + strlist[i][j] + "\n"
    strlist[i] = newstr

extracted_str = pd.DataFrame({'str' : strlist})
# extracted_str['str'] = extracted_str['str'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
extracted_str['str'].str.strip()
# print(extracted_str['str'])
#
extracted_str.to_excel('sample.xlsx')

df7 = pd.read_excel('./sample.xlsx', usecols=['str'])
# print(df7)