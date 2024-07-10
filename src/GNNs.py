#!/usr/bin/env python
# coding: utf-8

# # GNNs
# Similar to CNNs, which are able to exploit structural characteristics of images to improve their learning process, GNNs are suited for learning over *graph-like* data. Since Petri Nets carry essential information in their structure, using other models would not lead to useful results without some ingenious data engineering and feature extraction; GNNs allow for this process to be skipped almost entirely.
# 
# ## Creating a Hybrid System with GNNs
# For this project, we need to create a hybrid system, so using GNNs to learn and extract feature vectors from the Petri Nets is going to be the first step; these feature vectors can then be used by other models to infer characteristics from the original net. But first, we need to convert the data into a graph.
# 
# # PyTorch Geometric (PyG)
# PyTorch Geometric is a framework atop PyTorch for the construction of GNN models: it provides several layers from literature, which can then be combined for the creation of a GNN. But first, we have to transform the data a structure that PyG accepts.

# In[1]:


import torch_geometric.data as pyg_data


# In[ ]:


def get_homogeneous_data(Iterable):
    data = pyg_data.Data

