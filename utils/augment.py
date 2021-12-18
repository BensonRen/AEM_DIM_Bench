#!/usr/bin/env python
# coding: utf-8

# # Data distribution check

# In[14]:


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#import umap as umap

"""
# In[24]:


# Read the old and new data 
old = pd.read_csv('data_x_old.csv', header=None,sep=' ',dtype='float')
old.info()
old = old.values
new = pd.read_csv('data_x.csv', header=None,sep=' ',dtype='float')
new.info()
new = new.values


# ## Histogram

# In[29]:


# Plot the histogram of data
def histogram_plot(data, dim):
    f = plt.figure()
    # Determine if this is a new data
    if np.shape(data)[0] == 17500:
        new_flag = True
        name = 'new'
    else:
        new_flag = False
        name = 'old'
    # Plot the histogram
    plt.hist(data[:, dim],bins=100)
    plt.title('histogram of axim {} of {} data '.format(dim, name))
    plt.ylabel('cnt')
    plt.xlabel('axis {}'.format(dim))
    plt.savefig('histogram of axim {} of {} data.png'.format(dim, name))


# In[30]:


for i in range(8):
    histogram_plot(new, i)
    histogram_plot(old, i)


# ## Clustering

# In[31]:


data_all = np.concatenate([old, new])
reducer = umap.UMAP()
embedding = reducer.fit_transform(data_all)
embedding.shape


# In[37]:


# Plot the umap graph
lo = len(old)
ln = len(new)
label_all = np.zeros([lo + ln, ])
label_all[lo:] = 1
f = plt.figure()
plt.scatter(embedding[:lo, 0], embedding[:lo, 1], label='old',s=1)
plt.legend()
plt.xlabel('u1')
plt.ylabel('u2')
plt.title('umap plot for old data')
plt.savefig('umap plot for old data.png')
f = plt.figure()
plt.scatter(embedding[lo:, 0], embedding[lo:, 1], label='new',s=1)
plt.legend()
plt.xlabel('u1')
plt.ylabel('u2')
plt.title('umap plot for new data')
plt.savefig('umap plot for new data.png')
f = plt.figure()
plt.scatter(embedding[:lo, 0], embedding[:lo, 1], label='old',s=1)
plt.scatter(embedding[lo:, 0], embedding[lo:, 1], label='new',s=1)
plt.legend()
plt.xlabel('u1')
plt.ylabel('u2')
plt.title('umap plot for old data and new data')
plt.savefig('umap plot for old data and new data.png')


# ## Visualization
# 

# In[12]:


def plot_scatter(old, new, dim1, dim2):
    f = plt.figure()
    plt.scatter(old[:, dim1], old[:, dim2], label='old',marker='x')#,s=10)
    plt.scatter(new[:, dim1], new[:, dim2], label='new',marker='.')#,s=5)
    plt.legend()
    plt.xlabel('dim {}'.format(dim1))
    plt.ylabel('dim {}'.format(dim2))
    plt.title('scatter plot of dim{},{} of old and new data'.format(dim1, dim2))
    plt.savefig('scatter plot of dim{},{} of old and new data.png'.format(dim1, dim2))


# In[15]:


for i in range(8):
    for j in range(8):
        if i == j:
            continue
        plot_scatter(old, new, i, j)
        plt.close('all')


# ## Pair-wise scatter plot

# In[19]:


df_old = pd.DataFrame(old)
df_new = pd.DataFrame(new)
psm = pd.plotting.scatter_matrix(df_old, figsize=(15, 15), s=10)


# ## Find the same and plot spectra

# In[38]:


i = 0
for i in range(len(old)):
    #print(old[i,:])
    new_minus = np.sum(np.square(new - old[i,:]),axis=1)
    #print(np.shape(new_minus))
    match = np.where(new_minus==0)
    #print(match)
    if np.shape(match)[1] != 0: #There is a match
        print('we found a match! new index {} and old index {} match'.format(match, i))


# In[39]:


print('old index ', old[11819,:])
print('new index ', new[5444,:])


# In[35]:


np.shape(match)


# ### Plot the matched spectra

# In[6]:


y_old = pd.read_csv('data_y_old.csv',header=None,sep=' ')


# In[42]:


y_new = pd.read_csv('data_y_new.csv',header=None,sep=' ')


# In[7]:


y_old = y_old.values
y_new = y_new.values


# In[45]:


# plot the spectra
old_index = 11819
new_index = 5444
f = plt.figure()
plt.plot(y_old[old_index,:],label='old geometry {}'.format(old[old_index, :]))
plt.plot(y_new[new_index,:],label='new geometry {}'.format(new[new_index, :]))
plt.legend()
plt.ylabel('transmission')
plt.xlabel('THz')
plt.savefig('Spectra plot for identicle point')


# # Conclusion, this simulation is not the same as before ...

# ### See what percentage are still within range

# In[36]:


#print(old)
#print(new)
hmax = np.max(old[:,0])
hmin = np.min(old[:,1])
rmax = np.max(old[:,4])
rmin = np.min(old[:,4])

print(hmax, hmin, rmax, rmin)

#hmax = np.max(new[:,0])
#hmin = np.min(new[:,1])
#rmax = np.max(new[:,4])
#rmin = np.min(new[:,4])

#print(hmax, hmin, rmax, rmin)

within_range = np.ones([len(new)])

new_minus = np.copy(new)
new_minus[:,:4] -= hmin
new_minus[:,4:] -= rmin

new_plus = np.copy(new)
new_plus[:, :4] -= hmax
new_plus[:, 4:] -= rmax

small_flag = np.min(new_minus, axis=1) < 0
big_flag = np.max(new_plus, axis=1) > 0

within_range[small_flag] = 0
within_range[big_flag] = 0

print(np.sum(within_range) / len(within_range))
print(type(within_range))
print(np.shape(within_range))
print(within_range)
print(new[np.arange(len(within_range))[within_range.astype('bool')],:])
print(np.sum(within_range))


# # Data augmentation
# ## Since the geometry is symmetric, we can augment the data with permutations

# In[13]:


# Check the assumption that the permutation does indeed give you the same spectra
# Check if there is same spectra
i = 0
for i in range(len(y_old)):
    #print(old[i,:])
    new_minus = np.sum(np.square(y_old - y_old[i,:]),axis=1)
    #print(np.shape(new_minus))
    match = np.where(new_minus==0)
    #print(match)
    #print(np.shape(match))
    #print(len(match))
    #if match[0]
    if len(match) != 1:#np.shape(match)[1] != 0: #There is a match
        print('we found a match! new index {} and old index {} match'.format(match, i))


# ### Due to physical periodic boundary condition, we can augment the data by doing permutations

# In[39]:
"""

def permutate_periodicity(geometry_in, spectra_in):
    """
    :param: geometry_in: numpy array of geometry [n x 8] dim
    :param: spectra_in: spectra of the geometry_in [n x k] dim
    :return: output of the augmented geometry, spectra [4n x 8], [4n x k]
    """
    # Get the dimension parameters
    (n, k) = np.shape(spectra_in)
    # Initialize the output
    spectra_out = np.zeros([4*n, k])
    geometry_out = np.zeros([4*n, 8])
    
    #################################################
    # start permutation of geometry (case: 1 - 0123)#
    #################################################
    # case:2 -- 1032 
    geometry_c2 = geometry_in[:, [1,0,3,2,5,4,7,6]]
    # case:3 -- 2301
    geometry_c3 = geometry_in[:, [2,3,0,1,6,7,4,5]]
    # case:4 -- 3210
    geometry_c4 = geometry_in[:, [3,2,1,0,7,6,5,4]]
    
    geometry_out[0*n:1*n, :] = geometry_in
    geometry_out[1*n:2*n, :] = geometry_c2
    geometry_out[2*n:3*n, :] = geometry_c3
    geometry_out[3*n:4*n, :] = geometry_c4
    
    for i in range(4):
        spectra_out[i*n:(i+1)*n,:] = spectra_in
    return geometry_out, spectra_out


# In[40]:
data_folder = '/work/sr365/Christian_data/dataIn'
data_out_folder = '/work/sr365/Christian_data_augmented'
for file in os.listdir(data_folder):
    data = pd.read_csv(os.path.join(data_folder, file),header=None,sep=',').values
    (l, w) = np.shape(data)
    g = data[:,2:10]
    s = data[:,10:]
    g_aug, s_aug = permutate_periodicity(g, s)
    output = np.zeros([l*4, w])
    output[:, 2:10] = g_aug
    output[:, 10:] = s_aug
    np.savetxt(os.path.join(data_out_folder, file+'_augmented.csv'),output,delimiter=',')

# In[41]:


#print(np.shape(g))


# In[ ]:




