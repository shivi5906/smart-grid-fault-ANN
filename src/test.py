from graphviz import Digraph

dot = Digraph(format='png')

# Global styling
dot.attr(rankdir='LR', splines='curved')
dot.attr('node', shape='circle', style='filled', fontname='Helvetica')

# -------------------
# Input Layer
# -------------------
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(6):
        s.node(f'I{i}', f'I{i}', fillcolor='#A8DADC')  # light blue

# -------------------
# Hidden Layer 1
# -------------------
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(4):
        s.node(f'H1_{i}', '', fillcolor='#457B9D')  # darker blue

# -------------------
# Hidden Layer 2
# -------------------
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(3):
        s.node(f'H2_{i}', '', fillcolor='#1D3557')  # deep blue

# -------------------
# Output Layer
# -------------------
dot.node('O', 'Output', fillcolor='#E63946')  # red

# -------------------
# Connections (lighter edges)
# -------------------
for i in range(6):
    for j in range(4):
        dot.edge(f'I{i}', f'H1_{j}', color='gray', penwidth='0.5')

for i in range(4):
    for j in range(3):
        dot.edge(f'H1_{i}', f'H2_{j}', color='gray', penwidth='0.5')

for i in range(3):
    dot.edge(f'H2_{i}', 'O', color='gray', penwidth='0.5')

# -------------------
# Render
# -------------------
dot.render('nn_beautiful')