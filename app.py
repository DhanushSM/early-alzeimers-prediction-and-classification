from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file
import os
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json
import shutil
from model import AlzheimerModel

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii', 'nii.gz', 'dcm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'RESULTS_FOLDER': RESULTS_FOLDER,
    'MAX_CONTENT_LENGTH': MAX_FILE_SIZE
})

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize model
model = AlzheimerModel('alzheimer_model.h5')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image(img_path):
    try:
        img = Image.open(img_path)
        return model.predict_2d(img)
    except Exception as e:
        app.logger.error(f"Error analyzing image: {str(e)}")
        raise


def analyze_3d_scan(filepath):
    try:
        return model.predict_3d(filepath)
    except Exception as e:
        app.logger.error(f"Error analyzing 3D scan: {str(e)}")
        raise


def save_analysis_result(result, filename_prefix):
    result_file = os.path.join(RESULTS_FOLDER, f"{filename_prefix}_result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    return result_file


def generate_visualization(results, output_path):
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Only include classes that exist in your dataset
    class_names = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia']
    probabilities = results.get('probabilities', [0.7, 0.2, 0.1])[:3]  # Ensure only 3 values

    # Create colors - highlight the predicted class
    predicted_class = results.get('predicted_class', np.argmax(probabilities))
    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(len(class_names))]

    # Create the plot
    bars = ax.barh(class_names, probabilities, color=colors)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{width:.1%}',
                va='center', ha='left')

    ax.set_xlim(0, 1)
    ax.set_title('Diagnosis Confidence', fontweight='bold')
    ax.set_xlabel('Probability')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.content_length > MAX_FILE_SIZE:
            flash('File too large (max 50MB)')
            return redirect(request.url)

        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_id = str(uuid.uuid4())
                ext = filename.rsplit('.', 1)[1].lower()
                filename = f"{unique_id}.{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if ext in ['nii', 'nii.gz', 'dcm']:
                    results = analyze_3d_scan(filepath)
                    template = 'results_3d.html'
                else:
                    results = analyze_image(filepath)
                    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_plot.png")
                    generate_visualization(results, plot_path)
                    template = 'results_2d.html'

                # Ensure all required fields are present
                results['filename'] = filename
                results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if template == 'results_2d.html':
                    results['plot_filename'] = f"{unique_id}_plot.png"
                    # Ensure features exist and have default values if None
                    results['features'] = results.get('features', {
                        'hippocampal_atrophy': 0.0,
                        'ventricular_enlargement': 0.0,
                        'cortical_thinning': 0.0
                    })

                save_analysis_result(results, unique_id)

                return render_template(template, **results)

            except Exception as e:
                app.logger.error(f"Error processing file: {str(e)}", exc_info=True)
                flash('Error processing your scan. Please try again.')
                return redirect(request.url)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/results/<result_id>')
def view_result(result_id):
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}_result.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)

        if 'features' in results:  # 2D scan
            # Ensure features have default values if None
            for feature in ['hippocampal_atrophy', 'ventricular_enlargement', 'cortical_thinning']:
                if feature in results['features'] and results['features'][feature] is None:
                    results['features'][feature] = 0.0
            return render_template('results_2d.html', **results)
        else:  # 3D scan
            return render_template('results_3d.html', **results)
    else:
        flash('Result not found')
        return redirect(url_for('index'))


if __name__ == '__main__':
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)