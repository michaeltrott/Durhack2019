//
var imageArray = [];
var video;
var button;
function setup() {
  var canvas = createCanvas(600,600);
  video = createCapture(VIDEO);
  video.size(500,500);
  //Canvas.parent("canvasID");
  button = createButton("takePicture");
  button.mousePressed(takePic);
}
function takePic(){
	//save an image file
	imageArray.push(video.get());
	//adds the video image to the array
}

function draw() {
	var x = 0;
	var w = 100;
	var h = 50;
	for (var i=0;i<imageArray.length;i++){
		image(imageArray[i],x,0,w,h)
		
	}
  //background(200);
  //image(video,0,0,width,height);
}
