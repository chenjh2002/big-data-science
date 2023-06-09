from community import community_louvain
from dataset import dataset,real_label
import community

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx


# Load the Facebook network
G = nx.read_edgelist("facebook/0.edges")

# Compute the community partition using the Louvain algorithm
partition = community.best_partition(G)
modularity = community.modularity(partition, G)

# modularity
print(f"modularity: {modularity:05f}")

# NMI
from sklearn.metrics.cluster import normalized_mutual_info_score

# Convert the ground truth communities and the Louvain communities to lists
circles=real_label()
cur_len=len(circles)

true_labels = []
louvain_labels = []
for node_id, community_id in partition.items():
    louvain_labels.append(community_id)
    flag=0
    for circle_name, node_ids in circles.items():
        if node_id in node_ids:
            true_labels.append(circle_name)
            flag=1
            break
    
    if flag==0:
        true_labels.append(cur_len)
        cur_len+=1


# Compute the NMI between the ground truth communities and the Louvain communities
nmi = normalized_mutual_info_score(true_labels, louvain_labels)
print(f"NMI:{nmi:03f}")

import pandas as pd
pd.DataFrame({'Id':partition.keys(),'Group':partition.values()})


# draw the graph
pos = nx.spring_layout(G)

# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, 
                       pos, 
                       partition.keys(), 
                       node_size=40,
                       cmap=cmap, 
                       node_color=list(partition.values())
                      )
nx.draw_networkx_edges(G, pos, alpha=0.5)

plt.savefig('community.png')