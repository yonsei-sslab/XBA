import graphviz
import pandas
import json

# Opening JSON file
f = open("disasm.json")

# returns JSON object as
# a dictionary
disasm = json.load(f)

alignment_df = pandas.read_csv("alignment.csv", header=None)
labeled = list(alignment_df[0])
relation1_df = pandas.read_csv("gcn1-relation.csv", header=None)


def build_adjacency(relation):
    adj = {}
    for idx, row in relation.iterrows():
        a = row[0]
        re = row[1]
        b = row[2]

        neighbors = adj.get(a, [])
        adj[a] = neighbors + [str(b) + "-" + re]
        """
        neighbors = adj.get(b, [])
        adj[b] = neighbors + [a]
        """
    return adj


adj = build_adjacency(relation1_df)
dot = graphviz.Digraph(
    engine="twopi",
    graph_attr={"overlap": "compress", "ranksep": "0.8:1:1:1:1:2"},  # "0.8:1:1:1:1:2
    # graph_attr={"overlap": "false", "splines": "true"},
    # node_attr={"colorscheme": "set19"},
    # edge_attr={"colorscheme": "set19"},
)

seen = []
queue = [6667]
dot.node(str(6667), label="")
count = 0
while queue:
    count += 1
    if count > 200:
        break

    node = queue[0]
    queue = queue[1:]

    if node in seen:
        continue
    else:
        seen.append(node)

    neighbors = adj.get(node, [])
    for idx, neighbor in enumerate(neighbors):
        if idx > 20:
            break
        name, re = neighbor.split("-")

        if int(name) in labeled or not disasm[name]:
            color = "maroon"
        else:
            color = "navy"

        dot.node(
            name,
            label="",
            shape="circle",
            fillcolor=color,
            style="filled",
        )

        if re == "stringref":
            color = "green"
        elif re == "controlflow":
            color = "cornflowerblue"
        elif re == "calls":
            color = "chocolate4"
        elif re == "takesaddrof":
            color = "gold"
        else:
            raise NotImplemented

        dot.edge(str(node), str(name), color=color, penwidth="8", arrowhead="none")
        queue.append(int(name))

dot.render("doctest-output", view=True)
