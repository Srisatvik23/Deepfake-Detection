from flask import Flask, render_template, request
import os
from output import detect_deepfake

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        video = request.files['file']  # Fixed name to 'file'
        if video and allowed_file(video.filename):
            path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(path)
            result = detect_deepfake(path)
            os.remove(path)  # Clean up after processing
        else:
            result = "Invalid file format."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
