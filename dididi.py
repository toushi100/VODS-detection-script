import pickle,json
with open('objects.pickle','rb') as handle:
    b = pickle.load(handle)
    b = json.dumps(b)
    b = json.loads(b)
    print(b)