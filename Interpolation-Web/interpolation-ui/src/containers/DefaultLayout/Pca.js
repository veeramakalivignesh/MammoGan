import React, { Component } from 'react';
import { Button, Card, CardBody, Container, Col, CardHeader, Row,
  Spinner, Modal, ModalBody, ModalHeader } from 'reactstrap';
import axios from 'axios';
import Slider, { createSliderWithTooltip } from 'rc-slider';
import io from 'socket.io-client'
import 'rc-slider/assets/index.css';
import Annotator from "./Annotator";

const SliderWithTooltip = createSliderWithTooltip(Slider);

// const url = 'http://dsvmm275kcwn43rak.eastus.cloudapp.azure.com:8080/';
//const url = 'http://192.168.25.110:8080/';
const url = '/';

class CustomSlider extends Component{

  render(){
    const {pca,index} = this.props
    return(
      <div style = { { 'width': 700, 'height': 30, 'margin': 'auto', } }>
        <SliderWithTooltip min={-200} max={200} onChange = {(e)=> pca.state.on?pca.onChangeSlider(e,index):{}}
                            value = {pca.state.principleValues[index]}>Component {index}</SliderWithTooltip>
      </div>
    );
  }
} 

class Pca extends Component {

  constructor(props) {
    super(props);
    this.state = {
      uploading: false,
      realImageSrc : null,
      genImageSrc: null,
      genImageLoading: false,
      filePath : null,
      fileName : "",
      progress: null,
      isOpen : false,
      mask_canvas_state : null,
      principleValues : [0,0,0,0,0,0,0,0,0,0],
      on: false
    };
    this.socket = io.connect(url, { 'reconnection': false } );
    this.components = [0,1,2,3,4,5,6,7,8,9]
  }

  componentDidMount() {
    this.socket.on('connect', () => {
      console.log( "Socket Connection Established " + this.socket.id );
    } );
    this.socket.on('uploadMsg', (msg) => {
      this.setState( { progress: msg } );
    } );
  };

  uploadFormData = async (formData) => {
    return await axios({
      method: 'post',
      url: url + 'upload',
      data: formData,
      config: { headers: {'Content-Type': 'multipart/form-data' }}
    })
  };

  onImageUploadClick = () => {
    this.ImageUpload.click();
  };

  onImageUploadChange = async (e) => {
    const file = e.target.files[0];
    this.setState( {filePath: file, on: false} );
    let fileName = file.name.split(".")[0];
    this.setState({uploading: true, genImageLoading: true, progress: 'Uploading Image',
    fileName: fileName });
    let formData = new FormData();
    formData.append( 'socketId', this.socket.id );
    formData.append('file', file);
    try {
      const prefix = "data:image/png;base64,";
      const res = await this.uploadFormData(formData);
      this.setState( {
        realImageSrc: prefix + res.data.file,
        genImageSrc: prefix + res.data.file,
      } );
    } catch (e) {
      alert("An error occured while uploading " + fileName + " " + e);
    } finally {
      this.setState({ uploading: false, genImageLoading: false, progress: null });
    }
  };

  onChangeSlider = (e,i) => {
    var list = this.state.principleValues
    list[i] = e
    this.setState({
      principleValues: list,
      genImageLoading: true
    });
    this.socket.emit( 'moveSlider', {
        'alpha': e,
        'index': i
      }, (res) => {
        this.setState({
          genImageSrc : res ? "data:image/png;base64," + res : null,
          genImageLoading: false
        })
    } );
  };

  onReconstruct = async () => {
    if (!this.state.filePath) {
      alert("please choose an image");
      return;
    }
    this.setState({
      on: true
    });
    this.socket.emit( 'reconstruct', {}, (list,res) => {
      this.setState({
        principleValues: list, 
        genImageSrc: res ? "data:image/png;base64," + res : null,
        genImageLoading: false
      })
  } );
  };

  handleModalOpen = () => {
    this.setState( { isOpen: true } );
  };

  handleModalToggle = (maskCanvas) => {
    let maskCanvasState = maskCanvas.getSaveData();
    this.setState( { isOpen : !this.state.isOpen, mask_canvas_state: maskCanvasState } );
  };

  render() {
    return (
          <Container fluid>
          <Card>
            <CardHeader style={{textAlign: 'center'}}>
              <strong>Customizable Mammogram Toolkit</strong>
            </CardHeader>
            <Annotator isOpen = {this.state.isOpen} genImageSrc = {this.state.genImageSrc} url = {url}
            leftFileName = {this.state.leftFileName} rightFileName = {this.state.rightFileName}
            sliderValue = {this.state.sliderValue} handleModalToggle = {this.handleModalToggle}
                       maskCanvasState = {this.state.mask_canvas_state} />
            <CardBody>
              <Row>
              <Col sm xs="12" className="text-center mt-3">
                  <div style={{height: 512, width: 512, border : 'solid', margin: 'auto'}}>
                    { this.state.uploading ?
                      <Spinner animation="border" />
                      : < img src={this.state.realImageSrc} />  }
                  </div>
                </Col>
                <Col/>
                <Col sm xs="12" className="text-center mt-3">
                  <div style={{height: 512, width: 512, border : 'solid', margin : 'auto'}}>
                    { this.state.genImageLoading ? <Spinner animation="border" /> :
                      (this.state.genImageSrc ? < img src={this.state.genImageSrc} /> : null) }
                  </div>
                </Col>
                </Row>
                <Row>
                <Col sm xs="12" className="text-center mt-3">
                  {this.components.map(val => (
                    <CustomSlider pca = {this} index = {val}/>
                  ))}
                  <Button color="primary mt-1" onClick={ this.onReconstruct } > Reconstruct </Button>
                </Col>
                </Row>
              <Row>
              <Col sm xs="12" className="text-center mt-3">
                <Button color="primary mt-1" onClick={ this.onImageUploadClick } >
                    Upload Image
                    <input id = "realImage" ref = {e => this.ImageUpload = e}
                           type='file' onChange={ this.onImageUploadChange } hidden/>
                  </Button>
                </Col>
              </Row>
              <Row>
                <Col sm xs="12" className="text-center mt-3">
                  <Button color="primary mt-1" onClick={ this.handleModalOpen }  > Annotate Image </Button>
                </Col>
              </Row>
            </CardBody>
          </Card>
          </Container>
    );
  }
}

export default Pca;