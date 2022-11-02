import pandas as pd
import numpy as np

class WeightGetter():
    
    def __init__(self,df,cols = ['skin_tone','age','gender']):
        #currently because I'm lazy this assumes the class order is the same as the outputs
        #used for our prediction
        #used to get loss weights for model outputs of type [y_skintone, y_age, y_gender]
        #formatted to work with pytorch loss (after being converted to a tensor)
        if('is_face') in df.columns:
            df = df[df.is_face]
        self.weights = {c: self.class_weights(df,c) for c in cols}
        self.subgroup_weights = self.intersectional_class_weights(df,cols)
        
        
    def class_weights(self,df,col):
        percentages={}
        for c in df[col].unique():
            mean = (df[col] == c).mean()
            percentages[c] = mean
        weights = {c: 1/v for c,v in percentages.items()}
        max_w = np.max([v for k,v in weights.items()])

        w = np.zeros(len(df[col].unique()))
        for c,v in weights.items():
            w[int(c)] = v/max_w
        return w

    def intersectional_class_weights(self,df,cols):
        classes = [sorted(df[c].unique()) for c in cols]
        combinations = [[c] for c in classes[0]]
        for i,cset in enumerate(classes[1:]):
            pos = i + 1
            new_combinations = []
            for combination in combinations:
                for c in cset:
                    new_combo = combination[:] + [c]
                    new_combinations.append(new_combo)
            combinations = new_combinations

        percentages = []
        avg_w = 0
        for combination in combinations:
            valid = np.ones(df.shape[0]).astype(int)
            entry = {}
            for col, classval in zip(cols,combination):
                entry[col] = classval
                subset = (df[col] == classval).astype(int).ravel()
                valid = valid*subset
            entry['ratio'] = valid.mean()
            entry['weight'] = 1/valid.mean() if valid.mean() > 0 else 0
            avg_w += entry['weight'] / len(combinations)
            percentages.append(entry)

        for entry in percentages:
            entry['weight'] = entry['weight']/avg_w
    
        return percentages

    def get_weights(self,classname):
        #gets summary class weights for the whole group
        return self.weights.get(classname)
    
    def get_individual_weight(self,st,age,gender):
        #get loss weights in terms of subgroups
        w = self.subgroup_weights
        w = [i for i in w if i['skin_tone'] == int(st)]
        w = [i for i in w if i['age'] == int(age)]
        w = [i for i in w if i['gender'] == int(gender)]
        if len(w) > 1:
            print(w)
        if len(w) < 1:
            print('invalid class', 'st',st,'age',age,'gender',gender)
            return 1
        return w[0]['weight']