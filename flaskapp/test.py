import pickle
MODEL_PATH = r'D:\Sanskar\TelecoChurnMlendtoend\Notebook\churn_random_forest.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print(type(model))
