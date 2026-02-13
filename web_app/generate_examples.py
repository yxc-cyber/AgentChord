#!/usr/bin/env python3
"""
Generate example NetworkX graphs for testing the web visualizer.
"""
import os
import pickle
import random

import networkx as nx


def create_multiline_text_graph():
    """Create a graph with nodes containing multi-line text and newlines."""
    G = nx.Graph()
    
    # Add nodes with multi-line labels and long text
    nodes = [
        ("node1", {
            "label": "This is a very long text that should be wrapped into multiple lines automatically by the visualization system",
            "type": "long_text",
            "color": "#ff6b6b"
        }),
        ("node2", {
            "label": "Multi-line\nText Node\nWith Explicit\nLine Breaks",
            "type": "multiline",
            "color": "#4ecdc4"
        }),
        ("node3", {
            "label": "Short node",
            "type": "short",
            "color": "#45b7d1"
        }),
        ("node4", {
            "label": "Another very long piece of text that demonstrates how the system handles word wrapping and automatic line breaking for better readability",
            "description": "This node has additional\nmetadata with\nnewlines too",
            "type": "complex",
            "color": "#96ceb4"
        }),
        ("node5", {
            "label": "Mixed:\nThis node has both\nexplicit newlines and a very long sentence that should be wrapped automatically when displayed in the visualization",
            "type": "mixed",
            "color": "#ffeaa7"
        })
    ]
    
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    
    # Add edges connecting the nodes
    edges = [
        ("node1", "node2", {"label": "connects to\nmultiline", "weight": 1.0}),
        ("node2", "node3", {"label": "simple", "weight": 1.5}),
        ("node3", "node4", {"label": "leads to complex", "weight": 2.0}),
        ("node4", "node5", {"label": "mixed connection", "weight": 1.2}),
        ("node5", "node1", {"label": "cycles back", "weight": 0.8})
    ]
    
    for source, target, attrs in edges:
        G.add_edge(source, target, **attrs)
    
    return G


def create_simple_graph():
    """Create a simple undirected graph."""
    G = nx.Graph()
    
    # Add nodes with attributes
    nodes = [
        ("A", {"label": "Node A", "size": 20, "color": "#ff6b6b"}),
        ("B", {"label": "Node B", "size": 15, "color": "#4ecdc4"}),
        ("C", {"label": "Node C", "size": 25, "color": "#45b7d1"}),
        ("D", {"label": "Node D", "size": 18, "color": "#96ceb4"}),
        ("E", {"label": "Node E", "size": 22, "color": "#ffeaa7"})
    ]
    
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    
    # Add edges with labels
    edges = [
        ("A", "B", {"label": "connects", "weight": 1.5}),
        ("B", "C", {"label": "links to", "weight": 2.0}),
        ("C", "D", {"label": "related", "weight": 1.2}),
        ("D", "E", {"label": "associated", "weight": 1.8}),
        ("E", "A", {"label": "cycles to", "weight": 1.0}),
        ("A", "C", {"label": "shortcut", "weight": 3.0})
    ]
    
    for source, target, attrs in edges:
        G.add_edge(source, target, **attrs)
    
    return G
    """Create a simple undirected graph."""
    G = nx.Graph()
    
    # Add nodes with attributes
    nodes = [
        ("A", {"label": "Node A", "size": 20, "color": "#ff6b6b"}),
        ("B", {"label": "Node B", "size": 15, "color": "#4ecdc4"}),
        ("C", {"label": "Node C", "size": 25, "color": "#45b7d1"}),
        ("D", {"label": "Node D", "size": 18, "color": "#96ceb4"}),
        ("E", {"label": "Node E", "size": 22, "color": "#ffeaa7"})
    ]
    
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    
    # Add edges with labels
    edges = [
        ("A", "B", {"label": "connects", "weight": 1.5}),
        ("B", "C", {"label": "links to", "weight": 2.0}),
        ("C", "D", {"label": "related", "weight": 1.2}),
        ("D", "E", {"label": "associated", "weight": 1.8}),
        ("E", "A", {"label": "cycles to", "weight": 1.0}),
        ("A", "C", {"label": "shortcut", "weight": 3.0})
    ]
    
    for source, target, attrs in edges:
        G.add_edge(source, target, **attrs)
    
    return G

def create_directed_graph():
    """Create a directed graph representing a workflow."""
    G = nx.DiGraph()
    
    # Add nodes representing workflow steps
    nodes = [
        ("start", {"label": "Start", "type": "input", "color": "#2ecc71"}),
        ("process1", {"label": "Process Data", "type": "process", "color": "#3498db"}),
        ("decision", {"label": "Decision", "type": "decision", "color": "#f39c12"}),
        ("process2a", {"label": "Path A", "type": "process", "color": "#9b59b6"}),
        ("process2b", {"label": "Path B", "type": "process", "color": "#e74c3c"}),
        ("end", {"label": "End", "type": "output", "color": "#34495e"})
    ]
    
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    
    # Add directed edges representing workflow flow
    edges = [
        ("start", "process1", {"label": "begin", "weight": 1}),
        ("process1", "decision", {"label": "evaluate", "weight": 1}),
        ("decision", "process2a", {"label": "if true", "weight": 0.6}),
        ("decision", "process2b", {"label": "if false", "weight": 0.4}),
        ("process2a", "end", {"label": "complete", "weight": 1}),
        ("process2b", "end", {"label": "complete", "weight": 1})
    ]
    
    for source, target, attrs in edges:
        G.add_edge(source, target, **attrs)
    
    return G

def create_random_graph(n_nodes=10, edge_probability=0.3):
    """Create a random graph with the specified number of nodes."""
    G = nx.erdos_renyi_graph(n_nodes, edge_probability)
    
    # Add random attributes
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#dda0dd", "#98d8c8"]
    
    for node in G.nodes():
        G.nodes[node].update({
            "label": f"Node {node}",
            "size": random.randint(10, 30),
            "color": random.choice(colors),
            "value": random.randint(1, 100)
        })
    
    # Add edge weights
    for edge in G.edges():
        G.edges[edge].update({
            "weight": round(random.uniform(0.1, 3.0), 1),
            "type": random.choice(["strong", "weak", "medium"])
        })
    
    return G

def create_social_network():
    """Create a small social network graph."""
    G = nx.Graph()
    
    # Add people as nodes
    people = [
        ("Alice", {"age": 25, "job": "Engineer", "color": "#ff9ff3"}),
        ("Bob", {"age": 30, "job": "Designer", "color": "#54a0ff"}),
        ("Charlie", {"age": 28, "job": "Manager", "color": "#5f27cd"}),
        ("Diana", {"age": 26, "job": "Analyst", "color": "#00d2d3"}),
        ("Eve", {"age": 32, "job": "Developer", "color": "#ff6348"}),
        ("Frank", {"age": 29, "job": "Consultant", "color": "#2ed573"})
    ]
    
    for person, attrs in people:
        G.add_node(person, **attrs)
    
    # Add friendships as edges
    friendships = [
        ("Alice", "Bob", {"relationship": "friends", "years_known": 5}),
        ("Alice", "Charlie", {"relationship": "colleagues", "years_known": 3}),
        ("Bob", "Diana", {"relationship": "friends", "years_known": 7}),
        ("Charlie", "Eve", {"relationship": "siblings", "years_known": 28}),
        ("Diana", "Frank", {"relationship": "partners", "years_known": 4}),
        ("Eve", "Frank", {"relationship": "friends", "years_known": 2}),
        ("Alice", "Diana", {"relationship": "friends", "years_known": 6})
    ]
    
    for person1, person2, attrs in friendships:
        G.add_edge(person1, person2, **attrs)
    
    return G

def main():
    """Generate example graphs and save them as pickle files."""
    examples_dir = "example_graphs"
    os.makedirs(examples_dir, exist_ok=True)
    
    graphs = {
        "simple_graph.pkl": create_simple_graph(),
        "directed_workflow.pkl": create_directed_graph(),
        "random_graph.pkl": create_random_graph(15, 0.25),
        "social_network.pkl": create_social_network(),
        "multiline_text_graph.pkl": create_multiline_text_graph()
    }
    
    for filename, graph in graphs.items():
        filepath = os.path.join(examples_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        
        print(f"Created {filename}:")
        print(f"  - Nodes: {len(graph.nodes())}")
        print(f"  - Edges: {len(graph.edges())}")
        print(f"  - Type: {'Directed' if isinstance(graph, nx.DiGraph) else 'Undirected'}")
        print(f"  - Saved to: {filepath}")
        print()
    
    print(f"All example graphs saved in '{examples_dir}/' directory")
    print("You can now use these files to test the web visualizer!")

if __name__ == "__main__":
    main()