import time
s = time.time()

import os
import math

import bisect

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer








def pythonBinarySearch(l, element):
    
    index = bisect.bisect_left(l, element)
    
    if index < len(l) and l[index] == element:
        return True  # Element found
    else:
        return False  # Element not found





# Verified , Works Perfectly

def tokenizeSpeech ( stringOfSpeech ) :

    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    
    tokenList = tokenizer.tokenize(stringOfSpeech)
    
    return(tokenList)








def removeStopWordsAndReturnNewList (listOfTokens) :
    
    stopWordsList = stopwords.words('english')
    stopWordsList.sort()

    stopWordsRemovedListOfDocument = list()

    # Using binary search to check for stop words in list of tokens

    for token in listOfTokens:
                
        # if not (index < len(stopWordsList) and stopWordsList[index] == token):
        if not (pythonBinarySearch(stopWordsList , token)):
            stopWordsRemovedListOfDocument.append(token)

        
    return stopWordsRemovedListOfDocument










def applyPorterStemmerAndReturnNewList (r) :

    stemmer = PorterStemmer()    

    newList = list()

    for i in r:
        newList.append(stemmer.stem(i))

    newList.sort()

    return newList











def findTermFrequencyAndReturnDictionary (sortedStopWordsList) :
    d = dict()
    for i in range(len(sortedStopWordsList)):
        if i == 0:
            d[sortedStopWordsList[i]] = [1]
            continue
        if sortedStopWordsList[i] == sortedStopWordsList[i-1]:
            d[sortedStopWordsList[i]][0] = d[sortedStopWordsList[i]][0] + 1
        else:
            d[sortedStopWordsList[i]] = [1]
    return d










def applyLogTf (d) :
    import math
    for i in list(d.keys()):
        l = d[i]
        l.append(1 + math.log10(l[0]))
        d[i] = l
    return d










def populateDictionaryWithIDFValues (d) :
    import math
    
    allDocumentKeys = []
    for i in d:
        allDocumentKeys.append(i.keys())

    for i in range(len(d)):
        keysListOfCurrentDocument = d[i].keys()
        for key in keysListOfCurrentDocument:
            df = 0
            for k in range(len(d)):
                if pythonBinarySearch(list(d[k].keys()),key):
                    df += 1
            l = d[i][key]
            l.append(df)
            idf = math.log10(40/df)
            l.append(idf)
            tfidf = l[1] * l[3]
            l.append(tfidf)
            d[i][key] = l

    return d












def normalize_tfidf_values(d):
    
    for i in range(len(d)):
        norm = 0
        for j in d[i].keys():
            norm += (d[i][j][4] ** 2)
        norm = norm ** (1/2)

        for j in d[i].keys():
            l = d[i][j]
            l.append(l[4]/norm)
            d[i][j] = l
            # d[i][j][4] = (d[i][j][4]/norm)

    return d












def readAllFilesAndReturnNormalized_tfidf_dict ():
    
    d = []
    
    folder_path = './US_Inaugural_Addresses'
    
    for filename in os.listdir(folder_path):
    
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='windows-1252') as file:
                
                doc_content = file.read()
                doc_content = doc_content.lower()
                
                t = tokenizeSpeech(doc_content)
                s = removeStopWordsAndReturnNewList(t)
                p = applyPorterStemmerAndReturnNewList(s)
                tf = findTermFrequencyAndReturnDictionary(p)
                lg = applyLogTf(tf)
                
                d.append(lg)
            
    tfidf_dict = populateDictionaryWithIDFValues(d)
    normalized_tfidf_dict = normalize_tfidf_values(tfidf_dict)
    
    return normalized_tfidf_dict
#----------------------------------------------------------------------













def getidf (normalString) :
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    
    tokenList = tokenizer.tokenize(normalString)

    if len(tokenList) == 0:
        return -1

    # ----------------------------------------------------------
        
    stemmer = PorterStemmer()
    
    stemmedWord = stemmer.stem(tokenList[0])

    # ----------------------------------------------------------
    
    tokenListDocumentWise = []
    
    folder_path = './US_Inaugural_Addresses'
    
    for filename in os.listdir(folder_path):
        
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='windows-1252') as file:
                # Print the name of the file
                # print(f"Reading file: {filename
                # print(filename)
                
                doc_content = file.read()
                doc_content = doc_content.lower()
                
                t = tokenizeSpeech(doc_content)
                s = removeStopWordsAndReturnNewList(t)
                p = applyPorterStemmerAndReturnNewList(s)
                
                tokenListDocumentWise.append(p)

    df = 0
    
    for i in tokenListDocumentWise:
        if pythonBinarySearch (i, stemmedWord) :
            df += 1
        
    if df == 0:
        return -1
    else:
        return math.log10(40/df)














def preprocessToken (token):
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    
    tokenList = tokenizer.tokenize(token)
        
    stemmer = PorterStemmer()
    
    preprocessedToken = stemmer.stem(tokenList[0])

    return preprocessedToken













normalized_tfidf_dict = readAllFilesAndReturnNormalized_tfidf_dict()    

def getweight (filename,token):
    
    targetIndex = filename.split('_')
    if targetIndex[0][0] == "0":
        targetIndex = int(targetIndex[0][1]) - 1
    else:
        targetIndex = int(targetIndex[0]) - 1

    # pre process the token
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    
    tokenList = tokenizer.tokenize(token)

    if len(tokenList) == 0:
        # No tokenization possible
        # For example - 1234@#$()*
        return 0

        
    stemmer = PorterStemmer()
    
    preprocessedToken = stemmer.stem(tokenList[0])
    # ----------------------------------------------------------

    
    l = list(normalized_tfidf_dict[targetIndex].keys())
    
    if pythonBinarySearch(l , preprocessedToken):
        return normalized_tfidf_dict[targetIndex][preprocessedToken][5]
    else:
        return 0















def normalizeQueryVector (queryDict):
    norm = 0
    for i in list(queryDict.keys()):
        norm += (queryDict[i][1] ** 2)
    norm = norm ** (1/2)
        
    for i in list(queryDict.keys()):
        # norm += (queryDict[i][1] ** 2)
        l = queryDict[i]
        l.append(l[1]/norm)
        queryDict[i] = l

    return queryDict










def makeNormalizedQueryVector (q_document) :
    tokensList = tokenizeSpeech(q_document)
    s_w_r = removeStopWordsAndReturnNewList(tokensList)
    p = applyPorterStemmerAndReturnNewList(s_w_r)
    tf = findTermFrequencyAndReturnDictionary(p)
    ltf = applyLogTf(tf)
    n = normalizeQueryVector(ltf)
    return n












# Task 7.1 - Creating Posting List

postingDict = {}
for documentNumber in range(len(normalized_tfidf_dict)):
    for token in normalized_tfidf_dict[documentNumber].keys():
        if postingDict.get(token, False) == False:
            tf_idf_value = normalized_tfidf_dict[documentNumber][token][5]
            postingDict[token] = [[documentNumber , tf_idf_value]]
        else:
            tf_idf_value = normalized_tfidf_dict[documentNumber][token][5]
            l = postingDict[token]
            l.append([documentNumber , tf_idf_value])
            postingDict[token] = l











# Task 7.1 - Sorting Posting List in Decending Order

for i in list(postingDict.keys()):
    l = postingDict[i]
    sorted_data = sorted(l, key=lambda x: x[1], reverse=True)
    postingDict[i] = sorted_data













# Task 7.2 - return the top 10 documents based on the TF-IDF weights for each token in query.

# Task 7.2 - return the top 10 documents based on the TF-IDF weights for each token in query.


def query (queryString) :
    # ----------------------------------------------------------
    numberNonsense = False
    try:
        # print(queryString.split(" "))
        for i in queryString.split(" "):
            x = preprocessToken(i)
    except Exception as e:
        numberNonsense = True


    if numberNonsense == True:
        return f"( None , 0 )"

    # ----------------------------------------------------------
    
    
    l = makeNormalizedQueryVector(queryString)
    ll = list(l.keys())


    # -----------------------------------------------------
    uselessTokensInQuery = False
    for token in ll:
        if postingDict.get(token , False) == False:
            # print(token)
            uselessTokensInQuery = True
            break

    if uselessTokensInQuery == True:
        return f"( None , 0 )"
    # -----------------------------------------------------
    
    d = {}
    for token in ll:
        if postingDict.get(token , False) == False:
            continue
        else:
            d[token] = postingDict[token][0:10]
    
    setsList = []
    for key in d.keys():
        q = []
        for i in d[key]:
            q.append(i[0])
        setsList.append(set(q))
    
    # setsList
    commonDocx = set.intersection(*setsList)
    if len(commonDocx) == 0:
        return f"( Fetch More , 0 )"    
        
    # -----------------------------------------------------    
    
    
    storage = list()
    for documentNumber in commonDocx:
        d_temp = {}
        for key in d.keys():
            lll = d[key]
            for tf_idf_value_of_posting_dict in lll:
                if tf_idf_value_of_posting_dict[0] == documentNumber:
                    d_temp[key] = tf_idf_value_of_posting_dict[1]
                    break
        cosine = 0
        for key in d_temp.keys():
            cosine += (d_temp[key] * l[key][-1])
    
        storage.append([documentNumber+1 , cosine])
    
    
    maxIndex = 0
    cosine = 0
    for i in storage:
        if i[1] > cosine:
            maxIndex = i[0]
            cosine = i[1]

    # -----------------------------------------------------    

    
    fileNames = {
        1: '01_washington_1789.txt',
        2: '02_washington_1793.txt',
        3: '03_adams_john_1797.txt',
        4: '04_jefferson_1801.txt',
        5: '05_jefferson_1805.txt',
        6: '06_madison_1809.txt',
        7: '07_madison_1813.txt',
        8: '08_monroe_1817.txt',
        9: '09_monroe_1821.txt',
        10: '10_adams_john_quincy_1825.txt',
        11: '11_jackson_1829.txt',
        12: '12_jackson_1833.txt',
        13: '13_van_buren_1837.txt',
        14: '14_harrison_1841.txt',
        15: '15_polk_1845.txt',
        16: '16_taylor_1849.txt',
        17: '17_pierce_1853.txt',
        18: '18_buchanan_1857.txt',
        19: '19_lincoln_1861.txt',
        20: '20_lincoln_1865.txt',
        21: '21_grant_1869.txt',    
        22: '22_grant_1873.txt',
        23: '23_hayes_1877.txt',
        24: '24_garfield_1881.txt',
        25: '25_cleveland_1885.txt',
        26: '26_harrison_1889.txt',
        27: '27_cleveland_1893.txt',
        28: '28_mckinley_1897.txt',
        29: '29_mckinley_1901.txt',
        30: '30_roosevelt_theodore_1905.txt',
        31: '31_taft_1909.txt',
        32: '32_wilson_1913.txt',
        33: '33_wilson_1917.txt',
        34: '34_harding_1921.txt',
        35: '35_coolidge_1925.txt',
        36: '36_hoover_1929.txt',
        37: '37_roosevelt_franklin_1933.txt',
        38: '38_roosevelt_franklin_1937.txt',
        39: '39_roosevelt_franklin_1941.txt',
        40: '40_roosevelt_franklin_1945.txt'
    }
    
    
    return f"( File Name: {fileNames[maxIndex]} , Cosine: {cosine} )"
    
    # -----------------------------------------------------










# s = time.time()


print("--------------------------------------------")

print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))

print("--------------------------------------------")

print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))

print("--------------------------------------------")

print(query("states laws"))
print(query("war offenses"))
print(query("british war"))
print(query("texas government"))
print(query("world civilization"))

e = time.time()
print(f'\n\nTime taken for execution = {(e-s):.2f} Seconds')