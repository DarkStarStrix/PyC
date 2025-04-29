# /examples/simple_graph.pyc
class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def input(self, name):
        node = Node(name)
        self.nodes.append(node)
        return node

    def output(self, node):
        self.nodes.append(node)

    def __str__(self):
        return f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges"
    
class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []

    def __mul__(self, other):
        if isinstance(other, Node):
            edge = Edge(self, other)
            self.edges.append(edge)
            return Node(f"({self.name} * {other.name})")
        raise ValueError("Can only multiply Node with Node")

    def __add__(self, other):
        if isinstance(other, Node):
            edge = Edge(self, other)
            self.edges.append(edge)
            return Node(f"({self.name} + {other.name})")
        raise ValueError("Can only add Node to Node")
    
class Edge:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node

    def __str__(self):
        return f"Edge from {self.from_node.name} to {self.to_node.name}"
    
# Example usage
graph = Graph()
x = graph.input('x')
y = graph.input('y')
z = x * y
w = z + x
graph.output(w)

print(graph)
for node in graph.nodes:
    print(f"Node: {node.name}")
    for edge in node.edges:
        print(f"  {edge}")

