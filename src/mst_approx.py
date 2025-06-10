import networkx as nx

def mst_2_approx(coords):
    # Build complete graph
    G = nx.Graph()
    n = len(coords)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            dist = ((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)**0.5
            G.add_edge(i, j, weight=dist)
    # MST
    T = nx.minimum_spanning_tree(G)
    # Preorder traversal
    tour = []
    visited = set()
    def dfs(u):
        visited.add(u)
        tour.append(u)
        for v in sorted(T.neighbors(u)):
            if v not in visited:
                dfs(v)
    dfs(0)
    tour.append(0)
    return tour
