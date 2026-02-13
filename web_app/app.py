"""
Flask web application for visualizing NetworkX graphs from pickled files.
"""
import os
import pickle

import networkx as nx
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pkl', 'pickle'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def graph_to_dict(graph):
    """Convert NetworkX graph to dictionary format for JSON serialization."""
    # Convert graph to node-link format
    data = nx.node_link_data(graph)
    
    # Ensure node IDs are strings for consistency
    for node in data['nodes']:
        node['id'] = str(node['id'])
    
    for edge in data['links']:
        edge['source'] = str(edge['source'])
        edge['target'] = str(edge['target'])
    
    return data


@app.route('/')
def index():
    """Main page with file upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and redirect to visualization."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Try to load and validate the pickled graph
        try:
            with open(filepath, 'rb') as f:
                graph = pickle.load(f)
            
            if not isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph) and not isinstance(graph, nx.MultiGraph) and not isinstance(graph, nx.MultiDiGraph):
                flash('The uploaded file does not contain a valid NetworkX graph')
                os.remove(filepath)  # Clean up invalid file
                return redirect(url_for('index'))
            
            return redirect(url_for('visualize', filename=filename))
            
        except Exception as e:
            flash(f'Error loading the pickled file: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up invalid file
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload a .pkl or .pickle file')
        return redirect(url_for('index'))


@app.route('/visualize/<filename>')
def visualize(filename):
    """Display the graph visualization page."""
    return render_template('visualize.html', filename=filename)


@app.route('/api/graph/<filename>')
def get_graph_data(filename):
    """API endpoint to get graph data as JSON."""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        
        graph_data = graph_to_dict(graph)
        return jsonify(graph_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/path_input')
def path_input():
    """Page for inputting file path directly."""
    return render_template('path_input.html')


@app.route('/load_from_path', methods=['POST'])
def load_from_path():
    """Load graph from a file path."""
    file_path = request.form.get('file_path', '').strip()
    
    if not file_path:
        flash('Please enter a file path')
        return redirect(url_for('path_input'))
    
    if not os.path.exists(file_path):
        flash('File does not exist')
        return redirect(url_for('path_input'))
    
    try:
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        
        if not isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph) and not isinstance(graph, nx.MultiGraph) and not isinstance(graph, nx.MultiDiGraph):
            flash('The file does not contain a valid NetworkX graph')
            return redirect(url_for('path_input'))
        
        # Copy the file to uploads folder for consistency
        filename = f"path_loaded_{os.path.basename(file_path)}"
        if not filename.endswith(('.pkl', '.pickle')):
            filename += '.pkl'
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        # Copy the graph to our uploads folder
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        
        return redirect(url_for('visualize', filename=filename))
        
    except Exception as e:
        flash(f'Error loading the file: {str(e)}')
        return redirect(url_for('path_input'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)