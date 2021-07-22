import React, { Component } from 'react';
import { Button, Modal, ModalBody, ModalHeader } from 'reactstrap';
import CanvasDraw from "react-canvas-draw";
import '../../scss/annotator.scss'
import axios from 'axios';
import { saveAs } from 'file-saver';

class Annotator extends Component {

  constructor(props) {
    super(props);
    this.mask_canvas_ref = React.createRef();
    this.res_canvas_ref = null;
    this.b_width_slider = React.createRef();
    this.base_size = {width:512, height:512};
    this.is_mixing_gradients = false;
    this.btnStyle = {width: 50, height: 25,
      fontSize: 10, margin : '5px'};
    this.blend_position_offset = {
      x : 0,
      y : 0
    };
    this.state = {
      src_img: null,
      uploading: false,
      brushWidth : 5,
      erase : false,
      brushColor : "rgba(0,255,0,1.0)",
      maskFileName : ""
    };
    this.base_canvas_ref = document.createElement("canvas");
    this.src_canvas_ref = document.createElement("canvas");
    this.base_canvas_ref.width = this.src_canvas_ref.width = this.base_size.width;
    this.base_canvas_ref.height = this.src_canvas_ref.height = this.base_size.height;
  }

  onBaseImgChange = (genImageSrc) => {
    let base_canvas = this.base_canvas_ref;
    let base_ctx = base_canvas.getContext('2d');
    let result_canvas = this.res_canvas_ref;

    let result_ctx = result_canvas.getContext('2d');
    let base_size = this.base_size;
    let image = new Image();
    image.onload = function() {
      result_ctx.drawImage(image, 0, 0, base_size.width, base_size.height);
      base_ctx.drawImage(image, 0, 0, base_size.width, base_size.height);
    };
    image.src = genImageSrc;
  };

  uploadFormData = async (formData) => {
    return await axios({
      method: 'post',
      url: this.props.url + 'annotate',
      data: formData,
      config: { headers: {'Content-Type': 'multipart/form-data' }}
    })
  };

  onSrcImgChange = async (e) => {
    this.setState( { uploading: true } );
    let src_canvas = this.src_canvas_ref;
    let src_ctx = src_canvas.getContext('2d');
    let mask_ctx = this.mask_canvas_ref.current;
    mask_ctx.clear();
    const file = e.target.files[0];
    let base_size = this.base_size;

    let formData = new FormData();
    formData.append('file', file);

    try {
      const res = await this.uploadFormData(formData);
      const img_base64 = "data:image/png;base64," + res.data;
      this.setState( {
        src_img : img_base64
      } );
      let image = new Image();
        image.onload = function() {
          src_ctx.drawImage(image, 0, 0, base_size.width, base_size.height);
          mask_ctx.drawImage(image, 0, 0, base_size.width, base_size.height);
        };
        image.src = img_base64;
    } catch (e) {
      alert("An error occured while uploading");
    } finally {
      this.setState( { uploading : false } );
      this.setState( {maskFileName: file.name.split(".")[0] } )
    }

  };

  adjustBlendPosition = () => {
    let blend_position_offset = this.blend_position_offset;
    let src_canvas = this.src_canvas_ref;
    let src_ctx = src_canvas.getContext('2d');
    let base_size = this.base_size;
    let mask_canvas = this.mask_canvas_ref.current;
    let result_canvas = this.res_canvas_ref;
    let result_ctx = result_canvas.getContext('2d');
    let mask_ctx = mask_canvas.canvasContainer.children[1].getContext('2d');

    let src_pixels = src_ctx.getImageData(0, 0, base_size.width, base_size.height);
    let mask_pixels = mask_ctx.getImageData(0, 0, base_size.width, base_size.height);
    let result_pixels = result_ctx.getImageData(0, 0, base_size.width, base_size.height);

    for(let y=1; y<base_size.height-1; y++) {
      for(let x=1; x<base_size.width-1; x++) {
        let p = (y*base_size.width+x)*4;
        if( mask_pixels.data[p] === 0 && mask_pixels.data[p+1] === 255 &&
          mask_pixels.data[p+2] === 0 && mask_pixels.data[p+3] === 255) {
          let p_offseted = p + 4*((blend_position_offset.y)*base_size.width+blend_position_offset.x);
          for(let rgb=0; rgb<3; rgb++) {
            result_pixels.data[p_offseted+rgb] = src_pixels.data[p+rgb];
          }
        }
      }
    }
    result_ctx.putImageData(result_pixels, 0, 0);
  };


  moveBlendPosition = (direction) => {
    let base_size = this.base_size;
    let mask_canvas = this.mask_canvas_ref.current;
    let mask_ctx = mask_canvas.canvasContainer.children[1].getContext('2d');
    let mask_pixels = mask_ctx.getImageData(0, 0, base_size.width, base_size.height);
    let max = {x:base_size.width-2, y:base_size.height-2}, min = {x:0, y:0};

    if(direction === "up") {
      this.blend_position_offset.y-=10;
    } else if(direction === "right") {
      this.blend_position_offset.x+=10;
    } else if(direction === "down") {
      this.blend_position_offset.y+=10;
    } else if(direction === "left") {
      this.blend_position_offset.x-=10;
    }

    for(let y=1; y<base_size.height-1; y++) {
      for(let x=1; x<base_size.width-1; x++) {
        let p = (y*base_size.width+x)*4;
        if(mask_pixels.data[p]===0 && mask_pixels.data[p+1]===255 &&
          mask_pixels.data[p+2]===0 && mask_pixels.data[p+3]===255) {

          if((x+this.blend_position_offset.x)>max.x || (x+this.blend_position_offset.x)<min.x ||
            (y+this.blend_position_offset.y)>max.y || (y+this.blend_position_offset.y)<min.y) {

            if(direction === "up") {
              this.blend_position_offset.y+=10;
            } else if(direction === "right") {
              this.blend_position_offset.x-=10;
            } else if(direction === "down") {
              this.blend_position_offset.y-=10;
            } else if(direction === "left") {
              this.blend_position_offset.x+=10;
            }

            return false;
          }
        }
      }
    }

    let result_canvas = this.res_canvas_ref;
    let result_ctx = result_canvas.getContext('2d');
    let that = this;
    let image = new Image();
    image.onload = function() {
      result_ctx.drawImage(image, 0, 0, base_size.width, base_size.height);
      that.adjustBlendPosition();
    };
    image.src = this.props.genImageSrc;
  };

  blendImages = () => {

    let src_canvas = this.src_canvas_ref;
    let src_ctx = src_canvas.getContext('2d');
    let base_size = this.base_size;
    let mask_canvas = this.mask_canvas_ref.current;
    let result_canvas = this.res_canvas_ref;
    let result_ctx = result_canvas.getContext('2d');
    let mask_ctx = mask_canvas.canvasContainer.children[1].getContext('2d');
    let base_canvas = this.base_canvas_ref;
    let base_ctx = base_canvas.getContext('2d');
    let blend_position_offset = this.blend_position_offset;

    let base_pixels = base_ctx.getImageData(0, 0, base_size.width, base_size.height);
    let src_pixels = src_ctx.getImageData(0, 0, base_size.width, base_size.height);
    let mask_pixels = mask_ctx.getImageData(0, 0, base_size.width, base_size.height);
    let result_pixels = result_ctx.getImageData(0, 0, base_size.width, base_size.height);

    let dx, absx, previous_epsilon=1.0;
    let cnt=0;

    do {
      dx=0; absx=0;
      for(let y=1; y<base_size.height-1; y++) {
        for(let x=1; x<base_size.width-1; x++) {
          // p is current pixel
          // rgba r=p+0, g=p+1, b=p+2, a=p+3
          let p = (y*base_size.width+x)*4;

          // Mask area is painted rgba(0,255,0,1.0)
          if(mask_pixels.data[p]===0 && mask_pixels.data[p+1]===255 &&
            mask_pixels.data[p+2]===0 && mask_pixels.data[p+3]===255) {

            let p_offseted = p + 4*(blend_position_offset.y*base_size.width+blend_position_offset.x);

            // q is array of connected neighbors
            let q = [((y-1)*base_size.width+x)*4, ((y+1)*base_size.width+x)*4,
              (y*base_size.width+(x-1))*4, (y*base_size.width+(x+1))*4];
            let num_neighbors = q.length;

            for(let rgb=0; rgb<3; rgb++) {
              let sum_fq = 0;
              let sum_vpq = 0;
              let sum_boundary = 0;

              for(let i=0; i<num_neighbors; i++) {
                let q_offseted = q[i] + 4*(blend_position_offset.y*base_size.width+blend_position_offset.x);

                if(mask_pixels.data[q[i]+0]===0 && mask_pixels.data[q[i]+1]===255 &&
                  mask_pixels.data[q[i]+2]===0 && mask_pixels.data[q[i]+3]===255) {
                  sum_fq += result_pixels.data[q_offseted+rgb];
                } else {
                  sum_boundary += base_pixels.data[q_offseted+rgb];
                }

                if(this.is_mixing_gradients && Math.abs(base_pixels.data[p_offseted+rgb]-base_pixels.data[q_offseted+rgb]) >
                  Math.abs(src_pixels.data[p+rgb]-src_pixels.data[q[i]+rgb])) {
                  sum_vpq += base_pixels.data[p_offseted+rgb]-base_pixels.data[q_offseted+rgb];
                } else {
                  sum_vpq += src_pixels.data[p+rgb]-src_pixels.data[q[i]+rgb];
                }
              }
              let new_value = (sum_fq+sum_vpq+sum_boundary)/num_neighbors;
              dx += Math.abs(new_value-result_pixels.data[p_offseted+rgb]);
              absx += Math.abs(new_value);
              result_pixels.data[p_offseted+rgb] = new_value;
            }
          }
        }
      }
      cnt++;
      let epsilon = dx/absx;
      if(!epsilon || previous_epsilon-epsilon <= 1e-5) break; // convergence
      else previous_epsilon = epsilon;
    } while(true);
    result_ctx.putImageData(result_pixels, 0, 0);

    alert(cnt+" times iterated.");
  };

  onChangeBrushWidth = (e) => {
    this.setState( { brushWidth: parseInt( e.target.value ) } );
  };

  onClickDrawMode = () => {
    let mask_canvas = this.mask_canvas_ref.current;
    let mask_ctx = mask_canvas.canvasContainer.children[1].getContext('2d');
    mask_ctx.globalCompositeOperation = 'source-over';
    this.setState( { erase: false, brushColor : 'rgba(0,255,0,1.0)' } );
  };

  onClickEraseMode = () => {
    let mask_canvas = this.mask_canvas_ref.current;
    let mask_ctx = mask_canvas.canvasContainer.children[1].getContext('2d');
    mask_ctx.globalCompositeOperation = 'destination-out';
    this.setState( { erase: true, brushColor : 'black' } );
  };

  onClickSaveBlend = () => {
    let result_canvas = this.res_canvas_ref;
    const fileName = `L_${this.props.leftFileName}_R_${this.props.rightFileName}_Z_${this.props.sliderValue}.png`;
    result_canvas.toBlob(function(blob) {
      saveAs(blob, fileName);
    });
  };

  initDrawMask = () => {
    return (
      <div style={{ height: this.base_size.height + 100 }}>
        <h3>3. Select <span className="strong">Blend Source Image</span> &
          <span className="strong">Draw Mask</span></h3>
                <input type="file" onChange={ this.onSrcImgChange } />
        <div className="btn-group btn-group-sm" style={{height: 40}} role="group">
          <button type="button" style={ this.btnStyle } disabled={this.state.erase === false}
                  onClick={this.onClickDrawMode}
                  className="btn btn-secondary">Draw</button>
          <button type="button" style={ this.btnStyle } disabled={this.state.erase === true}
                  onClick={ this.onClickEraseMode }
                  className="btn btn-secondary">Erase</button>
        </div>
        <br/> <br/>
                <CanvasDraw canvasWidth={this.base_size.width}
                            canvasHeight={this.base_size.height}
                            ref={this.mask_canvas_ref}
                            imgSrc = {this.state.src_img}
                            brushColor = {this.state.brushColor}
                            brushRadius = {this.state.brushWidth}
                            saveData = { this.props.maskCanvasState }
                            immediateLoading = {true}
                            lazyRadius = {0}/>
          <span> Adjust Brush Width :
            <input ref={this.b_width_slider}
                                             type={"range"} onChange={this.onChangeBrushWidth}
           value={this.state.brushWidth} /> </span>
          <span className="btn-group btn-group-sm" style={{height: 40}} role="group">
          When done :
            <Button color="primary"
                             style = { this.btnStyle } onClick={this.adjustBlendPosition} >Click</Button>
            {/*<Button color="primary" style = { {width: 80, height: 25,*/}
              {/*fontSize: 10, margin : '5px' } } onClick={this.onClickSaveMask}*/}
                     {/*> <i className="fa fa-save"/> Save Mask</Button>*/}
          </span>

    </div>
    );
  };

  render() {

    let base_size = this.base_size;
    return (
      <Modal isOpen={this.props.isOpen} className={"modal-dialog modal-dialog-centered"}
             style = {{width: '50vw', maxWidth: '90vw'}} >
        <ModalHeader toggle={ () => this.props.handleModalToggle(this.mask_canvas_ref.current) } >Annotate</ModalHeader>
        <ModalBody style={{ maxHeight: 'calc(100vh - 100px)', overflowY: 'auto'}} >

      <div>
            <div id="front" className={"container"}>
              <div id="file-select" className={"clearfix"}>
                { this.initDrawMask() }
              </div>
              <br/> <br/> <br/> <br/>
              <div id="file-select" className={"clearfix"}>
                  <h3>4. Adjust <span className="strong">Blend Position</span></h3>
                  <canvas id="result-img" ref={ (r) => {
                    this.res_canvas_ref = r ;
                    if ( r ) this.onBaseImgChange( this.props.genImageSrc );
                    }
                  } width={base_size.width}
                          height={base_size.height} />
                  <div className="btn-group btn-group-sm" style={{height: 30}} role="group">
                    <button type="button" style={ this.btnStyle }
                            onClick={ () => { this.moveBlendPosition("left") } }
                            className="btn btn-primary">Left</button>
                    <button type="button" style={this.btnStyle}
                            onClick={ () => { this.moveBlendPosition("right") } }
                            className="btn btn-primary">Right</button>
                    <button type="button" style={this.btnStyle}
                            onClick={ () => { this.moveBlendPosition("down") } }
                            className="btn btn-primary">Down</button>
                    <button type="button" style={this.btnStyle}
                            onClick={ () => { this.moveBlendPosition("up") } }
                            className="btn btn-primary">Up</button>
                  </div>
                  <div className="mt-2" >
                  <Button color="primary" style = {{ width: 200, height : 30, fontSize: 15 }}
                       onClick={ this.blendImages }    > Start Blending </Button>
                  <Button color="primary ml-2" style = {{ width: 200, height : 30, fontSize: 15 }}
                        onClick={ this.onClickSaveBlend }    > <i className="fa fa-save"/> Save Blend </Button>
                  </div>
                </div>

            </div>
      </div>
        </ModalBody>
      </Modal>
    );
  }

}

export default Annotator;
