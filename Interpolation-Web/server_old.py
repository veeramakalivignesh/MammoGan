from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug import secure_filename
import os, base64
from flask_socketio import SocketIO, send, emit
import numpy as np
from encode_image import *
from PIL import Image
import eventlet

eventlet.monkey_patch()
session = {}

app = Flask(__name__, static_folder='interpolation-ui/build/static', template_folder='interpolation-ui/build')
cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins = "*", logger = True)

model = Model()
model.build_model()

@app.route('/')
def base():
    return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
@cross_origin()
def upload_file():
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
        left_z = np.expand_dims( left_z, 0 )
        right_z = np.expand_dims( right_z, 0 )
        
        session[ curr_sess_id ] = {
                'left_z': left_z,
                'right_z': right_z,
                'alpha': 0
         }

        #print ( left_z.shape )
        #print ( curr_sess_id )
        
        left_name, left_ext = left_file_name.split('.')
        right_name, right_ext = right_file_name.split('.')
        
        if left_ext != '.png':
            if os.path.exists( left_file_name ):
                os.remove( left_file_name )
            left_file_name = left_name + '.png'
        
        if right_ext != '.png':
            if os.path.exists( right_file_name ):
                os.remove( right_file_name )
            right_file_name = right_name + '.png'
            
        enc_img_left = open( left_file_name, 'rb' )
        left_encode = base64.b64encode( enc_img_left.read() )
        left_string = left_encode.decode('utf-8')

        enc_img_right = open( right_file_name, 'rb' )
        right_encode = base64.b64encode( enc_img_right.read() )
        right_string = right_encode.decode('utf-8')
        
        if os.path.exists( left_file_name ): 
            os.remove( left_file_name )
        if os.path.exists( right_file_name ):
            os.remove( right_file_name )
            
        json = jsonify( {
            'leftFile' : left_string, 
            'rightFile': right_string
        })
        return json

@socketio.on('connect')
def connect():
    print("Connection Established to", request.sid)
    global session
    curr_sess_id = request.sid
    if ( session.get( curr_sess_id ) is None ):
        session[ curr_sess_id ] = {}

@socketio.on('moveSlider')
def moveSlider(req):
    global session
    curr_sess_id = request.sid
    if ( session.get( curr_sess_id ) is None ):
        session[ curr_sess_id ] = {}
    info = session[ curr_sess_id ]
    info['alpha'] = req['alpha']
    alpha = req['alpha']
    print ("Generating Image for",alpha)
    if ( info.get( 'left_z' ) is None or info.get( 'right_z' ) is None ):
        return None
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
   
if __name__ == '__main__':
    socketio.run(app, port = 8080, host = '0.0.0.0')
    #app.run( port = 8080, host = '0.0.0.0', debug = False)
