import networkx as nx

def dataset():
    graph=nx.Graph()
    with open("./facebook/0.feat","r") as file:
        graph.add_nodes_from([i for i in range(1,len(file.readlines())+1)])
    
    with open("./facebook/0.edges","r") as file:
        edges=file.readlines()

        for edge in edges:
            start,end=edge.rstrip().split()
            graph.add_edge(int(start),int(end))
        
    return graph

def real_label():
    with open("facebook/0.circles", "r") as f:
        circles = {}
        for line in f:
            values = line.strip().split("\t")
            circle_name = values[0]
            node_ids = set(values[1:])
            circles[circle_name] = node_ids
    
    return circles


if __name__=='__main__':
    print(real_label())
