<!DOCTYPE html>
<html>
<head>
  <title>Algoritmo de Retropropagación con TensorFlow.js y Keras.js</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.8.6/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/kears@1.1.2/dist/keras.min.js"></script>
  <style>
    /* Mismo estilo que el código anterior */
  </style>
</head>
<body>
  <div class="container">
    <h1>Algoritmo de Retropropagación con TensorFlow.js y Keras.js</h1>
    <div class="form-group">
      <label for="x-input">Entradas (X):</label>
      <input type="text" id="x-input" placeholder="Formato: 1,2,3; 4,5,6" />
    </div>
    <div class="form-group">
      <label for="d-input">Salidas deseadas (d):</label>
      <input type="text" id="d-input" placeholder="Formato: 0,1,0" />
    </div>
    <div class="form-group">
      <label for="epochs-input">Épocas:</label>
      <input type="text" id="epochs-input" />
    </div>
    <div class="form-group">
      <label for="alpha-input">Tasa de aprendizaje (α):</label>
      <input type="text" id="alpha-input" />
    </div>
    <button onclick="trainModel()">Entrenar</button>
    <div id="result-label"></div>
  </div>

  <script>
    async function trainModel() {
      var xInput = document.getElementById("x-input").value;
      var dInput = document.getElementById("d-input").value;
      var epochsInput = document.getElementById("epochs-input").value;
      var alphaInput = document.getElementById("alpha-input").value;

      // Validar entradas (mismo código que antes)

      try {
        var X = tf.tensor2d(xInput.split(";").map(function(row) {
          return row.split(",").map(parseFloat);
        }));
        var d = tf.tensor2d([dInput.split(",").map(parseFloat)]);

        var model = tf.sequential();
        model.add(tf.layers.dense({ units: d.shape[1], inputShape: [X.shape[1]], activation: 'sigmoid' }));
        model.compile({ optimizer: tf.train.sgd(parseFloat(alphaInput)), loss: 'meanSquaredError' });

        await model.fit(X, d, { epochs: parseInt(epochsInput), verbose: 0 });

        var weights = model.getWeights();
        document.getElementById("result-label").textContent = "Pesos finales (W): " + weights[0].dataSync().join(", ") + "\nBias final (b): " + weights[1].dataSync()[0];
      } catch (error) {
        alert("Por favor, ingrese datos válidos. Asegúrese de que las entradas sean números.");
      }
    }
  </script>
</body>
</html>