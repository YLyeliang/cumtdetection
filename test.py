import pickle
# file=pickle.load(open('results.pkl','rb'))
CLASSES = ('water',)
a={cat: i + 1 for i, cat in enumerate(CLASSES)}
print(a)
debug=1
# print(file)