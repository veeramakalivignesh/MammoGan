from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import socketio 
import eventlet
import eventlet.wsgi
import numpy as np
from encode_image import *
from PIL import Image
from werkzeug.utils import secure_filename
import os, base64
from io import BytesIO
import sys

session = {}
ip = "0.0.0.0"

sio = socketio.Server(cors_allowed_origins = "*", async_mode='threading')
app = Flask(__name__, static_folder='interpolation-ui/build/static', template_folder='interpolation-ui/build')
cors = CORS(app)

cfg = get_cfg_defaults()
config_file = "../configs/mammogans_hd.yaml"
cfg.merge_from_file(config_file)

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.cuda.current_device()
print("Running on ", torch.cuda.get_device_name(device))

model = Model_Web(sio,cfg)
model.build_model()

@app.route('/')
def base():
    return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
@cross_origin()
def upload_file():
    with open('check.txt', 'w') as f:
        print('This message will be written to a file.', file=f)
    if request.method == 'POST':
        global session
        curr_sess_id = request.form['socketId']
        leftFile = request.files['leftFile']
        rightFile = request.files['rightFile']
        left_file_name = secure_filename(leftFile.filename)
        right_file_name = secure_filename(rightFile.filename)
        leftFile.save( left_file_name )
        rightFile.save( right_file_name ) 
        left_z, right_z = model.encode_image( left_file_name, right_file_name)
        # left_z = np.expand_dims( left_z.cpu(), 0 )
        # right_z = np.expand_dims( right_z.cpu(), 0 )
        
        session[ curr_sess_id ] = {
                'left_z': left_z,
                'right_z': right_z,
                'alpha': 0
         }

        left_name = left_file_name.split('.')[0]
        right_name = right_file_name.split('.')[0]
            
        enc_img_left = open( left_name + '.png', 'rb' )
        left_encode = base64.b64encode( enc_img_left.read() )
        left_string = left_encode.decode('utf-8')

        enc_img_right = open( right_name + '.png', 'rb' )
        right_encode = base64.b64encode( enc_img_right.read() )
        right_string = right_encode.decode('utf-8')
        
        del_files = [left_file_name, right_file_name, left_name + '.png', right_name + '.png']

        for f in del_files:
            if os.path.exists(f):
                os.remove(f)
            
        json = jsonify( {
            'leftFile' : left_string, 
            'rightFile': right_string
        })
        print(json)
        return json

@sio.on('connect')
def connect(sid, environ):
    print("Connection Established to", sid)
    global session
    curr_sess_id = sid
    if ( session.get( curr_sess_id ) is None ):
        session[ curr_sess_id ] = {}

@sio.on('moveSlider')
def moveSlider(sid, req):
    global session
    curr_sess_id = sid
    if ( session.get( curr_sess_id ) is None ):
        session[ curr_sess_id ] = {}
    info = session[ curr_sess_id ]
    info['alpha'] = req['alpha']
    alpha = req['alpha']
    #print ("Generating Image for", alpha)
    if ( info.get( 'left_z' ) is None or info.get( 'right_z' ) is None ):
        print ("No info")
        return ""
    left_z = info['left_z']
    right_z = info['right_z']
    model.get_gen_img(left_z, right_z, alpha)
    enc_img = open( 'gen.png', 'rb' )
    encode = base64.b64encode( enc_img.read() )
    string = encode.decode('utf-8')
    if os.path.exists( 'gen.png' ):
        os.remove( 'gen.png' )
    #print ("Done Generating") 
    return string

#app = socketio.Middleware(sio, app)
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)
app.run( port = 8080, host = ip, debug = False, threaded=True)
#eventlet.wsgi.server(eventlet.listen((ip, 8080)), app)
