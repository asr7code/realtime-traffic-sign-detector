const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const predictionText = document.getElementById("prediction");
const context = canvas.getContext("2d");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

// Send frame every 500ms
setInterval(() => {
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg");

  fetch("/predict", {
    method: "POST",
    body: JSON.stringify({ image: imageData }),
    headers: {
      "Content-Type": "application/json"
    }
  })
  .then(res => res.json())
  .then(data => {
    predictionText.innerText = "Prediction: " + data.prediction;
  });
}, 500);
