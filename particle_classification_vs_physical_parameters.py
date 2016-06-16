import sys, os
import matplotlib.pyplot as plt
import numpy as np
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)


net = caffe.Net("alex_ana.prototxt", 
                "/data/drinkingkazu/v03/singlep_alex/hires_filter/snapshot_rmsprop_iter_26000.caffemodel",
                caffe.TEST)

sorted_scores = [[], [], [], [], []]
for i in xrange(190):
    if i%10 == 0:
        print i
    net.forward()
    for j in xrange(100):
        sorted_scores[int(net.blobs['label'].data[j])].append(net.blobs['fc8'].data[j].copy())

kk = np.array([np.array(i) for i in sorted_scores])
for i in xrange(5):
    np.savetxt("%d.txt"%i, kk[i])

