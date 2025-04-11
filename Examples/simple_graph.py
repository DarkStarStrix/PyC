# /examples/simple_graph.pyc
graph = Graph()
x = graph.input('x')
y = graph.input('y')
z = x * y + 5
graph.output(z)
