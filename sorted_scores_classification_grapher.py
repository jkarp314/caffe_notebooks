import numpy as np
import matplotlib.pyplot as plt

sorted_scores = []
for i in xrange(5):
    sorted_scores.append(np.loadtxt("/home/jkarp314/sw/caffe_notebooks/%d.txt" % i,delimiter=' '))

sorted_classes = [[], [], [], [], []]
for i in xrange(5):
    for score in sorted_scores[i]:
        sorted_classes[i].append(score.argmax())
totals = np.zeros((5,5))
for i in xrange(5):
    for cl in sorted_classes[i]:
        totals[i][cl] += 1
fracs = np.empty((5,5))
for i in xrange(5):
    n = float(len(sorted_classes[i]))
    fracs[i] = totals[i]/n

f, plotarr = plt.subplots(5)
for i in range(5):
    plotarr[i].bar(range(5), fracs[i])
    plotarr[i].set_title('particle %d'%i)
    plotarr[i].set_ylim([0,1])
plt.show()

    
