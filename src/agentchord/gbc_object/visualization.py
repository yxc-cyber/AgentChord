
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

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


def plot_tree(G, pos=None, ax=None, node_size=None, save_path="gbc_tree_visualization.png",
              root=None, node_color="lightblue", width=16, height=8, horizontal_spacing=2.0, vertical_spacing=1.0, **kwargs):
    """
    Plot a tree with right-to-left orientation (root on right, children extending left).
    
    Parameters:
    -----------
    G : NetworkX graph
        The tree to visualize
    root : node, optional
        The root node to start from (will be detected if None)
    ax : matplotlib Axes object, optional
        Axes to draw on
    node_color : str or dict, optional
        Color of nodes or dictionary mapping node to color
    width : float
        Width of the figure
    height : float
        Height of the figure
    horizontal_spacing : float
        Spacing between hierarchy levels
    vertical_spacing : float
        Spacing between siblings
    **kwargs : additional arguments passed to networkx.draw_networkx_edges
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height))
    
    # If root is not specified, find it (assuming it's a tree)
    if root is None:
        # Root is likely the node with in-degree 0 or the lowest in-degree
        candidates = [n for n, d in G.in_degree() if d == 0]
        if candidates:
            root = candidates[0]
        else:
            # If no node with in-degree 0, pick the one with the least in-degree
            root = sorted(G.in_degree(), key=lambda x: x[1])[0][0]
    
    # Create a custom right-to-left layout
    pos = {}
    
    # Get all descendants by level
    def get_tree_levels(g, node, level=0, levels=None, visited=None):
        if levels is None:
            levels = {}
        if visited is None:
            visited = set()
            
        if node in visited:
            return levels
            
        visited.add(node)
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
        
        children = list(g.successors(node))
        for child in children:
            get_tree_levels(g, child, level + 1, levels, visited)
        return levels
    
    levels = get_tree_levels(G, root)
    max_level = max(levels.keys()) if levels else 0
    
    # Assign x coordinates (right to left)
    for level, nodes in levels.items():
        x = (max_level - level) * horizontal_spacing  # Right-to-left
        
        # Determine vertical placement for this level
        total_height = (len(nodes) - 1) * vertical_spacing
        start_y = -total_height / 2
        
        for i, node in enumerate(nodes):
            y = start_y + i * vertical_spacing
            pos[node] = (x, y)
    
    # Draw edges with curved arrows for better visibility
    curved_edges = [edge for edge in G.edges()]
    edge_color = kwargs.get('edge_color', 'black')
    
    # Draw edges with arrows pointing left
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=curved_edges, 
        arrows=True, 
        arrowstyle='->', 
        arrowsize=15, 
        edge_color=edge_color, 
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # Custom node drawing with rectangular boxes
    for node, (x, y) in pos.items():
        # Get node label - use node id if no label attribute exists
        if 'label' in G.nodes[node]:
            label = G.nodes[node]['label']
        else:
            label = str(node)
            
        # Split label into lines at existing '\n', then wrap each line to max 30 chars
        label_lines = []
        for raw_line in label.split('\n'):
            current_line = ""
            for word in raw_line.split():
                if len(current_line + " " + word) > 30 and current_line:
                    label_lines.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            if current_line:
                label_lines.append(current_line)
        multiline_label = '\n'.join(label_lines)
        
        # Calculate text size to determine box dimensions
        text = ax.text(
            x, y, multiline_label, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=node_color if not isinstance(node_color, dict) else node_color.get(node, 'lightblue'),
                edgecolor='black',
                alpha=0.9
            )
        )
        
        # Get the rendered text dimensions to ensure box fits
        renderer = plt.gcf().canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        width, height = bbox_data.width, bbox_data.height
        
        # Make box slightly larger than text
        box_width = width * 1.1
        box_height = height * 1.1
        
        # Remove the auto-generated text box
        text.remove()
        
        # Draw box manually
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle=f"round,pad=0.1",
            facecolor=node_color if not isinstance(node_color, dict) else node_color.get(node, 'lightblue'),
            edgecolor='black',
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Add text on top of the box
        ax.text(
            x, y, multiline_label, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=9
        )
    
    # Remove axis borders
    ax.set_axis_off()
    
    # Adjust plot to fit all nodes
    ax.margins(0.1)
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    


def visualize_gbc_tree(root: GBCBase, save_path="gbc_tree_visualization.png") -> None:
    """
    Visualizes the GBC object tree structure starting from the root node.
    
    Args:
        root (GBCBase): The root GBC object to visualize.
    """
    G = build_tree_graph(root)
    # draw_tree_on_root(G, root.get_node_representation(), save_path=save_path)
    plot_tree(G, save_path=save_path)