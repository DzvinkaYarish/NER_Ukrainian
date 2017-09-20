import pickle
import matplotlib.pyplot as plt
import numpy as np

f = open('../models/random_forest.pickle', "rb")
rf = pickle.load(f)
f.close()
features = ["Vocabulary Index", "IsUpper","POS", "Case", "Length", "Sentence Index"]
extended_features = []
for i in ['-1', '0', '1']:
    for f in features:
        extended_features.append(f + ' ' + i)


f_importances = sorted((list(zip(extended_features, rf.feature_importances_))), key=lambda x: x[1], reverse=True)
print([x[0] for x in f_importances])

# Plot the feature importances of the forest
width = 1
plt.figure()

x_pos = np.arange(len(extended_features))
plt.title("Feature importances")
plt.bar(x_pos, [x[1] for x in f_importances],
       color="r", align="center", width=width)
#plt.xticks(x_pos, [[x[0] for x in f_importances]])
plt.xlabel("features")

plt.xticks(x_pos, range(1, 19))
#plt.xlim([-1, 18])
plt.show()