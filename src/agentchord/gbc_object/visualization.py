
import matplotlib.pyplot as plt
import networkx as nx

from .gbc_base import GBCBase


def build_tree_graph(root: GBCBase) -> nx.DiGraph:
    G = nx.DiGraph()
    def add_edges(node):
        G.add_node(node.get_node_representation(), label=node.get_node_representation())
        for child in node.get_connections():
            if not isinstance(child, GBCBase):
                G.add_edge(node.get_node_representation(), str(child))
                G.add_node(str(child), label=str(child))
            else:
                G.add_edge(node.get_node_representation(), child.get_node_representation())
                add_edges(child)
    add_edges(root)
    return G


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Recursively assigns positions to nodes in a hierarchy (tree).
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    def _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter, pos, parent):
        children = list(G.successors(root))
        if not children:
            pos[root] = (xcenter, vert_loc)
        else:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, dx, vert_gap, vert_loc - vert_gap, nextx, pos, root)
            pos[root] = (xcenter, vert_loc)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter, {}, None)


def draw_tree_on_root(G, root, save_path="gbc_tree_visualization.png"):
    pos = hierarchy_pos(G, root)
    labels = nx.get_node_attributes(G, 'label')
    
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=1000, node_color="lightblue", edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)


def visualize_gbc_tree(root: GBCBase) -> None:
    """
    Visualizes the GBC object tree structure starting from the root node.
    
    Args:
        root (GBCBase): The root GBC object to visualize.
    """
    G = build_tree_graph(root)
    draw_tree_on_root(G, root.get_node_representation())