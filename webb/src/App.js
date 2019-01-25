import React, { Component } from 'react';
import './App.css';
import Webcam from "react-webcam";
import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

class App extends Component {
	constructor(props){
		super(props)

		this.state = {
			buttonText: ['Add glasses!', 'Remove glasses!'],
			orgImg: '',
			manImg: '',
			type: 0,
			epoch: 100
		}

		this.loadModel();
	}

	loadModel = async () => {
		const MODEL_URL = '/webcheckpoint'+ this.state.epoch +'/tensorflowjs_model.pb';
		const WEIGHTS_URL = '/webcheckpoint'+ this.state.epoch +'/weights_manifest.json';

		const fetchedModel = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);

		this.setState({
			model: fetchedModel
		});
	}

	setRef = webcam => {
		this.webcam = webcam;
	};

	capture = () => {
		const imageSrc = this.webcam.getScreenshot();

		var img = new Image;

		const onloadFunction = this.resizeImage;
		img.onload = function() {
			onloadFunction(img);
		};
		img.src = imageSrc;

		this.setState({
			orgImg: imageSrc,
			manImg: imageSrc
		});
	};

	changeType = () => {
		this.setState({
			type: this.state.type ? 0 : 1
		});
	};

	resizeImage = (image) => {
		var newDataUri = this.imageToDataUri(image, 128, 128);
		// continue from here...
		var img = new Image;

		let model = this.state.model;
		let updateState = this.updateImage;
		let imageIndex = this.state.type;
		img.onload = function() {
			let image = tf.fromPixels(this);
			image = image.reshape([1,128,128,3]);
			const newImage = model.execute({pred_0: image.toFloat(), pred_1: image.toFloat()});
			updateState(newImage[imageIndex].dataSync());
		};
		img.src = newDataUri;
	}

	updateImage = (image) => {
		image = image.map(function(elem){
			return (elem < 0) ? 0 : elem;
		});

		var c2 = document.createElement("canvas");
		var ctx2 = c2.getContext("2d");

		var c1 = document.createElement("canvas");
		c1.width = 128;
		c1.height = 128;
		var ctx1 = c1.getContext("2d");

		var imgData = ctx1.createImageData(128, 128);
		let added = 0;
		for (var i=0; i<imgData.data.length; i++) {
			if ((i+1) % 4 === 0) {
				imgData.data[i] = 255;
			}
			else {
				imgData.data[i] = image[added];
				added += 1;
			}
		}
		ctx1.putImageData(imgData, 0, 0);

		c2.width = 350;
		c2.height = 350;

		ctx2.mozImageSmoothingEnabled = false;
		ctx2.webkitImageSmoothingEnabled = false;
		ctx2.msImageSmoothingEnabled = false;
		ctx2.imageSmoothingEnabled = false;
		ctx2.drawImage(c1, 0, 0, 350, 350);

		this.setState({
			manImg: c2.toDataURL("image/jpeg")
		});
	}

	imageToDataUri = (img, width, height) => {
		// create an off-screen canvas
		var canvas = document.createElement('canvas'),
		ctx = canvas.getContext('2d');

		// set its dimension to target size
		canvas.width = width;
		canvas.height = height;

		// draw source image into the off-screen canvas:
		ctx.drawImage(img, 0, 0, width, height);

		// encode image to data-uri with base64 version of compressed image
		return canvas.toDataURL('image/jpeg', 1);
	}

	updateEpochs = (e) => {
		if (typeof e.target !== 'undefined') {
			this.setState({
				epoch: e.target.value
			}, () => {
				this.loadModel();
			});
		}
	}

	renderResults = () => {
		if (this.state.orgImg !== '') {
			return (
				<div className="flex items-center bg-grey-lighter mt-5 max-w-sm m-auto">
					<div className="flex-1 text-grey-darker text-center bg-grey-light px-4 py-2 m-2">
						Original:
						<img src={this.state.orgImg}/>
					</div>
					<div className="flex-1 text-grey-darker text-center bg-grey-light px-4 py-2 m-2">
						Manipulated:
						<img src={this.state.manImg}/>
					</div>
				</div>
			)
		}
	}

	render() {
		const videoConstraints = {
			width: 350,
			height: 350,
			facingMode: "user"
		};
		return (
			<div className="App">
				<header className="App-header">
					<h1>GLASSES ADDER/REMOVER</h1>
				</header>
				<div className="mt-5">
					<p>
						How many epochs:
						<select value={this.state.epoch} onChange={this.updateEpochs}>
							<option value="50">50</option>
							<option value="60">60</option>
							<option value="70">70</option>
							<option value="80">80</option>
							<option value="90">90</option>
							<option value="100">100</option>
						</select>
					</p>
					<button className={(!this.state.type ? 'bg-blue text-white' : 'bg-transparent border border-blue text-blue') +  " hover:text-white hover:bg-blue-dark font-bold py-2 px-4 rounded mr-2"} onClick={this.changeType}>Add glasses</button>
					<button className={(this.state.type ? 'bg-blue text-white' : 'bg-transparent border border-blue text-blue') +  " hover:text-white hover:bg-blue-dark font-bold py-2 px-4 rounded mb-5"} onClick={this.changeType}>Remove glasses</button>
					<br></br>
					<Webcam
						audio={false}
						height={350}
						ref={this.setRef}
						screenshotFormat="image/jpeg"
						width={350}
						videoConstraints={videoConstraints}
					/>
					<br></br>
					<button className="bg-blue hover:bg-blue-dark text-white font-bold py-2 px-4 rounded mt-4" onClick={this.capture}>{this.state.buttonText[this.state.type]}</button>
				</div>
				<div>
					{this.renderResults()}
				</div>
				<canvas id="myCanvas"/>
			</div>
		);
	}
}

export default App;
