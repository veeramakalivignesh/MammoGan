import React, { Component } from 'react';
import { Button, Card, CardBody, Container, Col, CardHeader, Row,
  Spinner, Modal, ModalBody, ModalHeader } from 'reactstrap';
import axios from 'axios';
import Slider, { createSliderWithTooltip } from 'rc-slider';
import io from 'socket.io-client'
import 'rc-slider/assets/index.css';

const SliderWithTooltip = createSliderWithTooltip(Slider);

// const url = 'http://dsvmm275kcwn43rak.eastus.cloudapp.azure.com:8080/';
//const url = 'http://192.168.25.110:8080/';
const url = '/';

class Interpolate extends Component {

  constructor(props) {
    super(props);
    this.state = {
      uploading: false,
      leftImageSrc : null,
      rightImageSrc: null,
      genImageSrc: null,
      genImageLoading: false,
      leftFilePath : null,
      rightFilePath: null,
      leftFileName : "",
      rightFileName : "",
      progress: null,
      sliderValue: 0,
      isOpen : false,
      mask_canvas_state : null
    };
    this.socket = io.connect(url, { 'reconnection': false } );
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

  onLeftImageUploadClick = () => {
    this.leftImageUpload.click();
  };

  onRightImageUploadClick = () => {
    this.rightImageUpload.click();
  };

  onLeftImageUploadChange = async (e) => {
    const file = e.target.files[0];
    this.setState( {leftFilePath: file} );
  };

  onRightImageUploadChange = async (e) => {
    const file = e.target.files[0];
    console.log(file)
    this.setState( {rightFilePath: file} );
  };

  onStart = async () => {
    if (!this.state.leftFilePath || !this.state.rightFilePath) {
      alert("Choose both images");
      return;
    }
    let leftFileName = this.state.leftFilePath.name.split(".")[0];
    let rightFileName = this.state.rightFilePath.name.split(".")[0];
    this.setState({uploading: true, genImageLoading: true, progress: 'Uploading Images',
    leftFileName: leftFileName, rightFileName: rightFileName });
    let formData = new FormData();
    formData.append( 'socketId', this.socket.id );
    formData.append('leftFile', this.state.leftFilePath);
    formData.append('rightFile', this.state.rightFilePath);
    try {
      const prefix = "data:image/png;base64,";
      const res = await this.uploadFormData(formData);
      this.setState( {
        leftImageSrc: prefix + res.data.leftFile,
        rightImageSrc: prefix + res.data.rightFile,
        genImageSrc: prefix + res.data.leftFile,
        sliderValue: 0
      } );
    } catch (e) {
      alert("An error occured while uploading " + leftFileName + " " + rightFileName + " " + e);
    } finally {
      this.setState({ uploading: false, genImageLoading: false, progress: null });
    }
  };

  onChangeSlider = (e) => {
    this.setState({
      sliderValue: e,
      genImageLoading: true
    });
    this.socket.emit( 'moveSlider', {
        'alpha': e
      }, (res) => {
        this.setState({
          genImageSrc : res ? "data:image/png;base64," + res : null,
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
            <CardHeader>
              <strong>Interpolation</strong>
            </CardHeader>
            <CardBody>
                <div style = { { 'width': 700, 'margin': 'auto',  } }>
                  <SliderWithTooltip onChange = {this.onChangeSlider}
                                     value = {this.state.sliderValue} />
                </div>
              <Row>
                <Col sm xs="12" className="text-center mt-3">
                  <p> {this.state.progress} </p>
                  <div style={{height: 512, width: 512, border : 'solid', margin : 'auto'}}>
                    { this.state.genImageLoading ? <Spinner animation="border" /> :
                      (this.state.genImageSrc ? < img src={this.state.genImageSrc} /> : null) }
                  </div>
                  <Button color="primary mt-1" onClick={ this.onStart } > Start </Button>
                </Col>
              </Row>
              <Row className="align-items-center mt-3">
                <Col sm xs="12" className="text-center mt-3">
                  <div style={{height: 512, width: 512, border : 'solid', margin: 'auto'}}>
                    { this.state.uploading ?
                      <Spinner animation="border" />
                      : < img src={this.state.leftImageSrc} />  }
                  </div>
                  <Button color="primary mt-1" onClick={ this.onLeftImageUploadClick } >
                    Upload Image
                    <input id = "leftImage" ref = { e => this.leftImageUpload = e}
                            onChange={ this.onLeftImageUploadChange }
                           type='file' hidden/>
                  </Button>
                </Col>
                <Col sm xs="12" className="text-center mt-3">
                  <div style={{height: 512, width: 512, border : 'solid', margin: 'auto'}}>
                    { this.state.uploading ?
                      <Spinner animation="border" />
                      : < img src={this.state.rightImageSrc} /> }
                  </div>
                  <Button color="primary mt-1" onClick={ this.onRightImageUploadClick } >
                    Upload Image
                    <input id = "rightImage" ref = {e => this.rightImageUpload = e}
                           type='file' onChange={ this.onRightImageUploadChange } hidden/>
                  </Button>
                </Col>
              </Row>
            </CardBody>
          </Card>
          </Container>
    );
  }
}

export default Interpolate;