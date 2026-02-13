# NetworkX Graph Visualizer

A web application for visualizing NetworkX graphs from pickled files with interactive drag-and-drop functionality.

## Features

- 📁 **File Upload**: Upload `.pkl` or `.pickle` files containing NetworkX graphs
- 📂 **Path Input**: Directly input file paths to graph files on your system
- 🖱️ **Interactive Visualization**: 
  - Drag nodes to rearrange them
  - Zoom and pan functionality
  - Edge labels displayed above connections
  - Node tooltips showing metadata
- � **Advanced Text Handling**:
  - Automatic text wrapping for long node labels
  - Support for explicit newlines (`\n`) in node text
  - Rectangle boxes around node content for better visibility
  - Multi-line text display with proper formatting
- �🔄 **Layout Controls**:
  - Refresh button to reset node positions
  - Fit to view functionality
  - Center graph button
- 📊 **Graph Information**: Display graph type, node count, and edge count
- 🎨 **Visual Features**:
  - Automatic node coloring based on ID
  - Support for directed and undirected graphs
  - Arrow markers for directed edges
  - Responsive design
  - Rectangular nodes with auto-sizing based on text content

## Installation

1. Navigate to the web_app directory:
```bash
cd web_app
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and go to:
```
http://localhost:5000
```

3. Choose one of the following options:
   - **Upload File**: Click "Choose File" and select a pickled NetworkX graph
   - **Input Path**: Enter the absolute path to your graph file

4. Once loaded, you can:
   - Drag nodes around to rearrange the graph
   - Use mouse wheel to zoom in/out
   - Drag the background to pan
   - Click "Refresh Layout" to reset node positions
   - Use "Fit to View" to auto-scale the graph
   - Hover over nodes to see their attributes

## Supported Graph Types

- `networkx.Graph` (undirected)
- `networkx.DiGraph` (directed) 
- `networkx.MultiGraph` (undirected multigraph)
- `networkx.MultiDiGraph` (directed multigraph)

## File Requirements

- Files must be pickled NetworkX graph objects
- Supported extensions: `.pkl`, `.pickle`
- Maximum file size: 16MB

## Graph Attributes

The visualizer will automatically display:
- **Node attributes**: All node attributes are shown in tooltips with proper newline handling
- **Text wrapping**: Long node labels are automatically wrapped at 50 characters per line
- **Newline support**: Explicit newlines (`\n`) in node labels are preserved and displayed correctly
- **Edge labels**: If edges have `label`, `weight`, or `key` attributes
- **Node colors**: Automatically generated based on node ID or custom colors from node attributes
- **Node sizes**: Rectangle size automatically calculated based on text content
- **Multi-line tooltips**: Attribute values with newlines are properly formatted in tooltips

## Troubleshooting

- **Import Error**: Make sure Flask and NetworkX are installed
- **File Not Found**: Check that the file path is correct and accessible
- **Invalid Graph**: Ensure the file contains a valid NetworkX graph object
- **Large Graphs**: For very large graphs, consider reducing the number of nodes or edges for better performance

## Development

The application uses:
- **Backend**: Flask (Python web framework)
- **Frontend**: D3.js (data visualization library)
- **Styling**: Bootstrap 5 (CSS framework)

To modify the visualization, edit the JavaScript code in `templates/visualize.html`.