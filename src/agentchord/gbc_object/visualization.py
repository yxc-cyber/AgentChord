import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle

from .gbc_base import GBCBase


def build_tree_graph(root: GBCBase) -> nx.DiGraph:
    G = nx.DiGraph()
    def add_edges(node):
        G.add_node(node.get_node_representation(), label=node.get_node_representation())
        for child, weight in zip(node.get_connections(), node.get_weights()):
            if not isinstance(child, GBCBase):
                G.add_edge(node.get_node_representation(), str(child), label=str(weight))
                G.add_node(str(child), label=str(child))
            else:
                G.add_edge(node.get_node_representation(), child.get_node_representation(), label=str(weight))
                add_edges(child)
    add_edges(root)
    return G


def plot_tree_right_to_left(G, root=None, ax=None, node_color='lightblue', 
                            min_width=16, min_height=8, **kwargs):
    """
    Plot a tree with right-to-left orientation with rectangles tightly fitting the text and edge labels above the lines.
    """
    # If root is not specified, find it
    if root is None:
        candidates = [n for n, d in G.in_degree() if d == 0]
        if candidates:
            root = candidates[0]
        else:
            root = sorted(G.in_degree(), key=lambda x: x[1])[0][0]
    
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
    
    # Calculate node content sizes
    node_boxes = {}
    for node in G.nodes():
        if 'label' in G.nodes[node]:
            label = G.nodes[node]['label']
        else:
            label = str(node)
        
        # Split long text into multiple lines (limit to 50 chars per line)
        label_lines = []
        current_line = ""

        for word in label.split():
            if len(current_line + " " + word) > 50 and current_line:
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
        
        # Create a temporary text object to measure rendered size
        text = plt.text(0, 0, multiline_label, fontsize=9, ha='center', va='center')
        renderer = plt.gcf().canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer)
        text.remove()  # Remove the temporary text object
        
        # Calculate width and height based on rendered text size
        width = bbox.width / plt.gcf().dpi  # Convert from pixels to inches
        height = bbox.height / plt.gcf().dpi
        
        node_boxes[node] = {
            'label': multiline_label,
            'width': width,
            'height': height
        }
    
    # Calculate appropriate figure dimensions
    max_nodes_per_level = max(len(nodes) for nodes in levels.values())
    
    # Determine spacing based on content size
    avg_width = np.mean([box['width'] for box in node_boxes.values()])
    avg_height = np.mean([box['height'] for box in node_boxes.values()])
    
    horizontal_spacing = max(2.0, avg_width * 3)
    vertical_spacing = max(1.0, avg_height * 4)
    
    # Calculate figure size
    width = max(min_width, (max_level + 1) * horizontal_spacing * 1.5)
    height = max(min_height, max_nodes_per_level * vertical_spacing * 1.5)
    
    # Create plot if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height))
    
    # Create a custom right-to-left layout
    pos = {}
    
    # Assign positions based on levels
    for level, nodes in levels.items():
        x = (max_level - level) * horizontal_spacing  # Right-to-left
        
        total_height = (len(nodes) - 1) * vertical_spacing
        start_y = -total_height / 2
        
        for i, node in enumerate(nodes):
            y = start_y + i * vertical_spacing
            pos[node] = (x, y)
    
    # Draw nodes first
    for node, (x, y) in pos.items():
        # Get node size and label
        box = node_boxes[node]
        width, height, multiline_label = box['width'], box['height'], box['label']
        
        # Draw rectangle manually
        rect = Rectangle(
            (x - width / 2, y - height / 2),  # Bottom-left corner
            width, height,  # Width and height
            facecolor=node_color if not isinstance(node_color, dict) else node_color.get(node, 'lightblue'),
            edgecolor='black',
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add text on top of the rectangle
        ax.text(
            x, y, multiline_label, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=9,
            zorder=3
        )
        
        # Store box dimensions for edge connections
        node_boxes[node].update({
            'x': x,
            'y': y,
            'left': x - width / 2,
            'right': x + width / 2,
            'top': y + height / 2,
            'bottom': y - height / 2
        })
    
    # Get edge labels from the graph
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    # Draw edges
    edge_color = kwargs.get('edge_color', 'black')
    arrow_style = kwargs.get('arrowstyle', '->')
    
    for parent, child in G.edges():
        # Get box information
        parent_box = node_boxes[parent]
        child_box = node_boxes[child]
        
        # Calculate connection points
        start_x = parent_box['left']
        start_y = parent_box['y']
        end_x = child_box['right']
        end_y = child_box['y']
        
        # Draw arrow
        ax.annotate(
            '', xy=(end_x, end_y), xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle=arrow_style,
                color=edge_color,
                lw=1.5
            ),
            zorder=1
        )
        
        # Add edge label above the connecting line
        if (parent, child) in edge_labels:
            label = edge_labels[(parent, child)]
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(
                mid_x, mid_y + 0.05, label,  # Slightly above the line
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='bottom',
                color='black',
                zorder=4
            )
    
    # Remove axis borders
    ax.set_axis_off()
    
    # Expand limits to show all nodes with padding
    x_min = min([box['left'] for box in node_boxes.values()])
    x_max = max([box['right'] for box in node_boxes.values()])
    y_min = min([box['bottom'] for box in node_boxes.values()])
    y_max = max([box['top'] for box in node_boxes.values()])
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    return ax


def visualize_gbc_tree(root: GBCBase, save_path="gbc_tree_visualization.png") -> None:
    """
    Visualizes the GBC object tree structure starting from the root node.
    
    Args:
        root (GBCBase): The root GBC object to visualize.
    """
    G = build_tree_graph(root)
    plot_tree_right_to_left(G)
    plt.savefig(save_path)