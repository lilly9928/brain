import sys

import pandas as pd
import re

# original data location
# data_src = 'D:/data/brain/brain_hemo_214.xlsx'
# term_src = '/term.xlsx'

# kbs data location-------------------------
data_src = 'C:/Users/andlabkbs/Desktop/meddataset/202cases/brain_hemo_214.xlsx'
term_src = './../term.xlsx'
#-------------------------------------------

df = pd.read_excel(data_src, skiprows=5, usecols=['ID', 'readout'])
df = df.dropna()
df.reset_index(drop=False, inplace=True)

wordlist = []
wordoutlist = ['ct', 'fracture', 'eye', 'dental']

# column에 있는 word를 wordlist에 추가
def add_wordlist(columnname):
    df = pd.read_excel(term_src, usecols=[columnname])
    df[columnname] = df[columnname].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df = df.dropna()  # NaN 값 제거
    df.reset_index(drop=False, inplace=True)  # NaN 값 제거 후 처음부터 인덱스 다시 부여
    for i in range(len(df)):
        x = df[columnname][i].split()
        for j in range(len(x)):
            if x[j] not in wordlist:
                wordlist.append(x[j].lower())

def add_wordoutlist(columnname):
    df = pd.read_excel(term_src, usecols=[columnname])
    df[columnname] = df[columnname].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    df = df.dropna()  # NaN 값 제거
    df.reset_index(drop=False, inplace=True)  # NaN 값 제거 후 처음부터 인덱스 다시 부여
    for i in range(len(df)):
        x = df[columnname][i].split()
        for j in range(len(x)):
            if x[j] not in wordoutlist:
                wordoutlist.append(x[j].lower())

add_wordoutlist('Otorhinolaryngology')

add_wordlist('hemo')
add_wordlist('Key Brain Terms Glossary')

strlist = []
IDlist = []

def check_date_format(input_date):
	regex = r'\d{4}-\d{2}-\d{2}'
	return  bool(re.findall(regex, input_date))

def check_num_front(input):
    regex=r'\d'
    return re.match(regex, input)

def isKorean(text):
    hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    result = hangul.findall(text)
    return len(result)

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
                check = 1 # 들어가면 안되는 단어 있는지 확인
        for k in range(len(wordlist)):
            if wordlist[k] in no_special_str.lower():
                num = num + 1
        if len(no_special_str) > 0 and no_special_str[0].isnumeric() :
            no_special_str = no_special_str[1:]
        if num > 0 and check == 0 and isKorean(no_special_str) == 0:
            tmplist.append(no_special_str)

    IDlist.append(df['ID'][i])
    strlist.append(tmplist)

for i in range(len(strlist)):
    newstr = ""
    for j in range(len(strlist[i])):
        newstr = newstr + strlist[i][j] + "\n"
    strlist[i] = newstr

extracted_str = pd.DataFrame(zip(IDlist, strlist), columns=['ID','str'])
extracted_str['str'].str.strip()
extracted_str.to_excel('sample.xlsx')