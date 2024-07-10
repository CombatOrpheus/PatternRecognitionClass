#!/usr/bin/env python
# coding: utf-8

# # Parsing the Data
# The data set is available in a JSON format, in both original and pre-processed formats. Since my needs are different from that of the original authors, I'll be using the original data. The data is organised as follows:
# - `data{index}`
#   - **petri_net**: A compound matrix $[I; O; M_0]$ representing the Petri Net; for a Petri Net with $N$ transitions, the compound matrix will have a length of $N*2 + 1$.
#   - **arr_vlist**: The reachable markings of this Petri Net.
#   - **arr_edge**: The edges (source, destination) for the reachable markings.
#   - **arr_tranidx**: The transition that fired, leading to the creation of a new marking.
#   - **spn_labda**: The $\lambda$ (average firing rate) of the transition.
#   - **spn_steadyprob**: The steady probability distribution of the Petri Net.
#   - **spn_markdens**: The token probability density function of the Petri Net.
#   - **spn_mu**: The average number of tokens in the network in the steady state.
# 
# For my purposes, only the first four data points are of interest, so I'll focus on extracting them. These correspond to the *Petri Net* and the *Reachability Graph*.

# ## Imports
# Due to the size of the files and the number of examples, it is more efficient to iterate through the values and build individual values. Python's standard JSON module is not adequate for this because it has to read the entire file before parsing, which is wasteful. Instead, `ijson` offers a steaming alternative, which is what I will be using.

# In[1]:


import json
import json_stream
import numpy
from pathlib import Path


# ### Typing Information
# Since I don't like to guess the return of a function based on its name, typing information is going to be used.

# In[2]:


from typing import Iterable, Dict


# # Reading the Data
# The data is already arranged in a readable format, which means that we simply have to parse the JSON structure.
# ## The First Solution: Default Python Module
# Python offers a JSON decoder/encoder in its standard library, and it might prove sufficient for our needs. The only issue with it, at first, is that it loads the entire file into memory before reading, which might slow down the process considerably for some of the larger files.

# In[3]:


def get_data_dict_blocking(file: Path) -> Dict:
    with open(file) as source:
        return json.load(source)


# ### Partial Solution: Asynchronous Reading
# Python's `asyncio` module offers high-level operations for asynchronous operations, particularly geared towards IO, which is exactly what I am doing. This does not solve the entire file still needs to be loaded into memory and then parsed, which is still a slow operation, considering that no other concurrent work is going to happen at the same time.

# In[4]:


async def get_data_dic_async(file: Path) -> Dict:
    with open(file) as source:
        return json.load(source)


# ## A Second Attempt: Chunked Reading
# Since blocking the entire process while reading and decoding the file is not ideal, iterating over the data, decoding item by item, and converting them concurrently should provide a reasonable performance improvement.
# 
# The `json_stream` library offers this feature transparently, although we do need to pay attention due to its iterator-based nature. We can only read the file once, and we need to store the results to use them. This is not too problematic for us since multiple conversions are not needed.

# In[5]:


def get_data_dict_stream(file: Path) -> Dict:
    with open(file) as source:
        return json_stream.load(source)


# ### The Third Solution: Improving the data
# The original data is a single giant JSON list, which makes decoding expensive. A simple solution is to simply reformat the data, so that each line has a single JSON element, eliminating the list. This way, I can read the data in a streaming fashion without any weird tricks.

# In[6]:


def get_data_line_iterator(file: Path) -> Iterable[Dict]:
    with open(file, 'rb') as source:
        for line in source:
            yield json.loads(line)


# # Converting the Data
# The data is read as a bunch of Python lists, which is not adequate for feeding into Neural Networks. For that end, I have to provide functions to convert this data into Numpy arrays, which are then trivially converted into PyTorch/TensorFlow Tensors.
# Additionally, I have to convert the Petri Net into a graph, so let's start with that.

# In[7]:


def get_petri_graph(pn: numpy.array):
    num_places, num_transitions = pn.shape
    # The number of transitions is (columns-1)/2, so we can simply round down.
    num_transitions = num_transitions // 2
    places = list(range(num_places))
    transitions = (list(range(num_places, num_places+num_transitions)))

    # Find the edges
    # place -> transition
    pt_edges = numpy.argwhere(pn[:, 0:num_transitions])
    # transition -> place
    tp_edges = numpy.argwhere(pn[:, num_transitions:-1])

    return (places, transitions, pt_edges, tp_edges)


# With this, I can create both homogeneous graphs and heterogeneous; the latter are probably more adequate, given that there is a real distinction between places and transitions in the net, and their links also have a different meaning. With this done, we can now convert the rest of the data.

# In[8]:


def get_petri_nets(source: Iterable) -> Iterable:
    for data in source:
        pn = numpy.array(data['petri_net'])
        reachable_markings = numpy.array(data['arr_vlist'])
        edges = numpy.array(data['arr_edge'])
        fired_transitions = numpy.array(data['arr_tranidx'])
        yield (pn, reachable_markings, edges, fired_transitions)

