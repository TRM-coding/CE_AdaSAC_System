import pickle

with open('GPTJ_ISP_VAL.pkl','rb') as f:
    dict=pickle.load(f)
print(dict)