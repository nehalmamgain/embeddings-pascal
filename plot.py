import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
import os

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io
from tqdm import tqdm

input_dir = 'pascal_emb'

sem_emb = [] # list of 825 300x64 embeddings on unseen pascal
vis_emb = []
vis_emb_orig = []
sem_lab = []
vis_lab = []
enc_learnt = []
enc_orig = []
enc_label = []

       
with open(os.path.join(input_dir, 'semantic_emb.file'), 'rb') as f:
    while True:
        try:
            sem_emb.append(pickle.load(f))
        except EOFError:
            break
print 'Got semantic embeddings'
'''
with open(os.path.join(input_dir, 'visual_emb.file'), 'rb') as f:
    while True:
        try:
            vis_emb.append(pickle.load(f))
        except EOFError:
            break
print 'Got visual embeddings'
with open(os.path.join(input_dir, 'visual_emb_orig_space.file'), 'rb') as f:
    while True:
        try:
            vis_emb_orig.append(pickle.load(f))
        except EOFError:
            break
print 'Got original visual embeddings'
'''
with open(os.path.join(input_dir, 'semantic_labels.file'), 'rb') as f:
    while True:
        try:
            sem_lab.append(pickle.load(f))
        except EOFError:
            break
print 'Got semantic labels'
'''
with open(os.path.join(input_dir, 'visual_labels.file'), 'rb') as f:
    while True:
        try:
            vis_lab.append(pickle.load(f))
        except EOFError:
            break
print 'Got visual labels'
'''
with open(os.path.join(input_dir, 'norm_enc_learnt_emb.file'), 'rb') as f:
    while True:
        try:
            enc_learnt.append(pickle.load(f))
        except EOFError:
            break
print 'Got encoded learnt embeddings'
with open(os.path.join(input_dir, 'norm_enc_orig_emb.file'), 'rb') as f:
    while True:
        try:
            enc_orig.append(pickle.load(f))
        except EOFError:
            break
print 'Got encoded orig embeddings'
with open(os.path.join(input_dir, 'joint_labels.file'), 'rb') as f:
    while True:
        try:
            enc_label.append(pickle.load(f))
        except EOFError:
            break
print 'Got encoded learnt embeddings'

print len(sem_emb)
print sem_emb[0].shape
print len(sem_lab)
print sem_lab[0].shape
print len(enc_learnt)
print enc_learnt[0].shape
print len(enc_orig)
print enc_orig[0].shape
print len(enc_label)
print enc_label[0].shape


#print len(vis_emb)
#print vis_emb[0].shape
print 'Final DS lengths'

RS = 123

# Utility function to visualize the outputs of PCA and t-SNE

iteri = 0
def scatter(x, colors, model):
    '''
    print type(x)
    print x.shape
    print type(colors)
    print colors.shape
    '''
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    #print x[:,0], '\n', x[:,1]
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    #print 'Here 1'
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)-1])
    #print 'Here 2'
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    #print 'Here 3'

    # add the labels for each digit corresponding to the label
    #print 'Here 4'

    txts = []
    '''
    lab_map = {0: 'background', 1: 'car', 2: 'dog', 3: 'sofa', 4: 'train'}
    for i in range(1, num_classes+1): # start from 0 for background and till num_classes

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
	print num_classes, ' ', xtext, ' ', ytext
        txt = ax.text(xtext, ytext, lab_map[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
	print txts
        txts.append(txt)
    '''
    #'''
    # set title
    plt.title(model+" embeddings PASCAL 15 epochs", fontsize=22)
    
    # legend with color patches
    patch_0 = mpatches.Patch(color=palette[0], label='background')
    patch_1 = mpatches.Patch(color=palette[1], label='car')
    patch_2 = mpatches.Patch(color=palette[2], label='dog')
    patch_3 = mpatches.Patch(color=palette[3], label='sofa')
    patch_4 = mpatches.Patch(color=palette[4], label='train')

    #leg = plt.legend(loc='center left', bbox_to_anchor(1,0.815), numpoints=1)
    #fig.savefig(outfile, bbox_extra_artists=(leg,), bbox_inches='tight')

    #plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], fontsize=12, loc=4)
    leg = plt.legend(handles=[patch_0, patch_1, patch_2, patch_3, patch_4], fontsize=14, loc='center left', bbox_to_anchor=(1,0.815), numpoints=1)
    #'''
    #print 'Here 5'

    global iteri
    iteri += 1
    #plt.savefig('plot'+str(iteri)+'.png')
    plt.savefig('newplot'+str(iteri)+'.png', bbox_extra_artists=(leg,), bbox_inches='tight')
    #print 'Here 6'

    return f, ax, sc, txts

sem_emb_arr = np.asarray(sem_emb)
x_sem = sem_emb_arr.reshape((sem_emb_arr.shape[0]*sem_emb_arr.shape[1],sem_emb_arr.shape[2]))
print x_sem.shape

vis_emb_arr = np.asarray(vis_emb)
x_vis = vis_emb_arr.reshape((vis_emb_arr.shape[0]*vis_emb_arr.shape[1],vis_emb_arr.shape[2]))
print x_vis.shape

vis_emb_orig_arr = np.asarray(vis_emb_orig)
x_vis_orig = vis_emb_orig_arr.reshape((vis_emb_orig_arr.shape[0]*vis_emb_orig_arr.shape[1],vis_emb_orig_arr.shape[2]))
print x_vis_orig.shape

sem_lab_arr = np.asarray(sem_lab)
y_sem = sem_lab_arr.reshape((sem_lab_arr.shape[0]*sem_lab_arr.shape[1],))
print y_sem.shape

vis_lab_arr = np.asarray(vis_lab)
y_vis = vis_lab_arr.reshape((vis_lab_arr.shape[0]*vis_lab_arr.shape[1],))
print y_vis.shape

unique, counts = np.unique(y_sem, return_counts = True)
counts_sem = dict(zip(unique,counts))
print counts_sem

list_del = []
for i, item in enumerate(y_sem):
    if item == 0:
	#print 'sem ', i
	list_del.append(i)

#print len(list_del), ' ' , int(np.mean(counts[1:]))
# comment below line to completely remove background class
list_del = np.random.choice(list_del, len(list_del)-int(np.mean(counts[1:])), replace=False)
#print len(list_del)

x_sem = np.delete(x_sem, list_del, axis=0)
y_sem = np.delete(y_sem, list_del, axis=0)

unique, counts = np.unique(y_sem, return_counts = True)
counts_sem = dict(zip(unique,counts))
print counts_sem

print x_sem.shape, ' ', y_sem.shape

vec_sem_orig = dict()
embedding_file = 'apascal_train_test.txt'#'apascal_test.txt'
unseen_classes = set(['__background__','car','dog','sofa','train'])
unseen_classes_map = {'__background__': 0,'car':1,'dog':2,'sofa':3,'train':4}
classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'table', 'dog', 'horse',
                 'motorbike', 'person', 'plant',
                 'sheep', 'sofa', 'train', 'television']

with io.open(embedding_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('Reading word vecs...')
        for line in tqdm(lines):
                line = line.split()
		if line[1] in unseen_classes:
			value = np.array([float(c) for c in line[6:]])
                	if unseen_classes_map[line[1]] in vec_sem_orig.keys(): 
				vec_sem_orig[unseen_classes_map[line[1]]].append(value)
			else:
				vec_sem_orig[unseen_classes_map[line[1]]] = [value]
print 'vec_sem_orig lengths ', len(vec_sem_orig), ' ', len(vec_sem_orig[1]), ' ', len(vec_sem_orig[2]), ' ', len(vec_sem_orig[3]), ' ', len(vec_sem_orig[4])

# Find bg vector (average across each category for min points)
min_points = len(vec_sem_orig[1])
for key in vec_sem_orig.keys():
	if len(vec_sem_orig[key]) < min_points:
		min_points = len(vec_sem_orig[key])

list_bg = []		
for i in range(min_points):
	bg_new = np.zeros(len(vec_sem_orig[1][0]))
	for key in vec_sem_orig.keys():
		bg_new += vec_sem_orig[key][i]
	bg_new /= len(vec_sem_orig)
	list_bg.append(bg_new)

vec_sem_orig[0] = list_bg
print 'vec_sem_orig lengths ', len(vec_sem_orig), ' ', len(vec_sem_orig[0])

# Convert dict of lists of embeddings (vec_sem_orig) to x_sem_orig, y_sem_orig
x_sem_orig = np.asarray(vec_sem_orig[0])
y_sem_orig = np.zeros(len(vec_sem_orig[0]),dtype=int)
print 'x_sem_orig and y_sem_orig ', x_sem_orig.shape, ' ', y_sem_orig.shape

for key in vec_sem_orig.keys():
	if key != 0:
		list_arr = np.asarray(vec_sem_orig[key])
		x_sem_orig = np.vstack((x_sem_orig, list_arr))
		for j in range(len(vec_sem_orig[key])):
			y_sem_orig = np.append(y_sem_orig, np.array([key]), axis=0)

print 'x_sem_orig and y_sem_orig ', x_sem_orig.shape, ' ', y_sem_orig.shape

unique, counts = np.unique(y_vis, return_counts = True)
counts_vis = dict(zip(unique,counts))
print counts_vis

list_del = []
for i, item in enumerate(y_vis):
    if item == 0:
	#print 'vis ', i
	list_del.append(i)

#print len(list_del), ' ' , int(np.mean(counts[1:]))
# comment below line to completely remove background class
list_del = np.random.choice(list_del, len(list_del)-int(np.mean(counts[1:])), replace=False)
#print len(list_del)

x_vis = np.delete(x_vis, list_del, axis=0)
x_vis_orig = np.delete(x_vis_orig, list_del, axis=0)
y_vis = np.delete(y_vis, list_del, axis=0)

unique, counts = np.unique(y_vis, return_counts = True)
counts_vis = dict(zip(unique,counts))
print counts_vis

print x_vis.shape, ' ', y_vis.shape, ' ', x_vis_orig.shape

print 'Semantic space'
print 'PCA for dim. reduction...'
time_start = time.time()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_sem)

print 'PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start)

print 'Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_))

print 't-SNE...'
time_start = time.time()

sem_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

print "Shapes of sem_pca_tsne and y_sem:", sem_pca_tsne.shape, ' ', y_sem.shape
scatter(sem_pca_tsne, y_sem, 'Semantic')


print 'Original semantic space'
print 'PCA for dim. reduction...'
time_start = time.time()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_sem_orig)

print 'PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start)

print 'Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_))

print 't-SNE...'
time_start = time.time()

sem_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

print "Shapes of sem_pca_tsne and y_sem:", sem_pca_tsne.shape, ' ', y_sem_orig.shape
scatter(sem_pca_tsne, y_sem_orig, 'Original semantic')


print 'Visual space'
print 'PCA for dim. reduction...'
time_start = time.time()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_vis)

print 'PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start)

print 'Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_))

print 't-SNE...'
time_start = time.time()

vis_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

scatter(vis_pca_tsne, y_vis, 'Visual')


print 'Original visual space'
print 'PCA for dim. reduction...'
time_start = time.time()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_vis_orig)

print 'PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start)

print 'Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_))

print 't-SNE...'
time_start = time.time()

vis_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

scatter(vis_pca_tsne, y_vis, 'Original visual')
