from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import socketio 
import eventlet
import eventlet.wsgi
import numpy as np
from encode_image_pca import *
from PIL import Image, ImageOps
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
# config_file = "../configs/mammogans_512.yaml"
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
        file = request.files['file']
        file_name = secure_filename(file.filename)
        file.save(file_name)
        z = model.encode_image(file_name)
        
        session[ curr_sess_id ] = {
                'z_orig': z,
                'z': z,
                'principleValues': [0,0,0,0,0,0,0,0,0,0]
         }

        name = file_name.split('.')[0]
        
        img = Image.open(name + '.png')
        img = ImageOps.grayscale(img)
        img = np.asarray(img)
        if img.shape[0]!=img.shape[1]:
            img = make_square(img)
        img = img.astype('uint8')
        img = Image.fromarray(img)
        img = img.resize((256,256))
        img = img.resize((512,512),Image.BICUBIC)
        # img = img.resize((512,512))
        img.save('resized.png')

        enc_img = open('resized.png', 'rb')
        encode = base64.b64encode( enc_img.read() )
        string = encode.decode('utf-8')
        
        del_files = ['resized.png']

        for f in del_files:
            if os.path.exists(f):
                os.remove(f)
            
        json = jsonify( {
            'file' : string, 
        })

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
    if (info.get( 'z' ) is None):
        print ("No info")
        return ""
    z = info['z']
    i = req['index']

    new_latents = model.get_new_latents(z,i,alpha-info['principleValues'][i])
    info['z'] = new_latents
    info['principleValues'][i] = alpha

    model.get_gen_img(new_latents)
    enc_img = open( 'gen.png', 'rb' )
    encode = base64.b64encode( enc_img.read() )
    string = encode.decode('utf-8')

    # if os.path.exists( 'gen.png' ):
    #     os.remove( 'gen.png' )

    return string

@sio.on('reconstruct')
def reconstruct(sid,req):
    global session
    curr_sess_id = sid
    if ( session.get( curr_sess_id ) is None ):
        session[ curr_sess_id ] = {}
    info = session[ curr_sess_id ]
    if (info.get( 'z' ) is None):
        print ("No info")
        return ""
    z = info['z_orig']
    info['z'] = z

    list = model.get_components(z)
    info['principleValues'] = list
    model.get_gen_img(z)
    enc_img = open( 'gen.png', 'rb' )
    encode = base64.b64encode( enc_img.read() )
    string = encode.decode('utf-8')

    # if os.path.exists( 'gen.png' ):
    #     os.remove( 'gen.png' )

    return (list,string)

@app.route('/annotate', methods = ['GET', 'POST'])
@cross_origin()
def annotate_src():
    if request.method == 'POST':
        src_img = request.files['file']
        img_file = secure_filename(src_img.filename)
        src_img.save( img_file )
        image = Image.open(img_file)
        image = model.resize(image, model.img_size)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode( buffered.getvalue() )

        if os.path.exists( img_file ):
            os.remove( img_file )
               
        return img_str

#app = socketio.Middleware(sio, app)
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)
app.run( port = 8080, host = ip, debug = False, threaded=True)
#eventlet.wsgi.server(eventlet.listen((ip, 8080)), app)
