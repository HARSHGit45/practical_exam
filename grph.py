def print_solution(color):
    print("Solution Exists: Following are the assigned colors")
    print(" ".join(map(str, color)))

def is_safe(v, graph, color, c):
    for i in range(len(graph)):
        if graph[v][i] and c == color[i]:
            return False
    return True

def graph_coloring_util(graph, m, color, v):
    # Base case: If all vertices are assigned a color
    if v == len(graph):
        return True

    # Consider colors 1 to m
    for c in range(1, m + 1):
        if is_safe(v, graph, color, c):
            color[v] = c  # Assign the color

            # Recursively assign colors to the next vertex
            if graph_coloring_util(graph, m, color, v + 1):
                return True

            color[v] = 0  # Backtrack

    return False

def graph_coloring(graph, m):
    color = [0] * len(graph)  # Initialize all vertices as uncolored (0)

    if not graph_coloring_util(graph, m, color, 0):
        print("Solution does not exist")
        return False

    print_solution(color)
    return True

# Driver code
# Input explanation:
# V: Number of vertices
# Graph: Adjacency matrix of the graph (1 for edge, 0 for no edge)
# m: Number of colors
V = int(input("Enter the number of vertices: "))  # Number of vertices
graph = []
print("Enter the adjacency matrix row by row:")

for i in range(V):
    row = list(map(int, input().split()))  # Input adjacency matrix
    graph.append(row)

m = int(input("Enter the number of colors: "))  # Input number of colors

graph_coloring(graph, m)




v =4 




    /* Create following graph and test
       whether it is 3 colorable
      (3)---(2)
       |   / |
       |  /  |
       | /   |
      (0)---(1)
    */
    bool graph[V][V] = {
        { 0, 1, 1, 1 },
        { 1, 0, 1, 0 },
        { 1, 1, 0, 1 },
        { 1, 0, 1, 0 },
    };
 
    // Number of colors
    int m = 3;

