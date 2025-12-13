/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Network
*/

#include "Network.hpp"

/**
 * @brief Construct a new Network:: Network object
 */
Network::Network()
    : _inputBuffer(1, 400), _hiddenBuffer(1, 128), _policyLogitsBuffer(1, 400),
      _valueOutBuffer(1, 1), _weights1(400, 128), _biases1(1, 128),
      _weights2(128, 400), _biases2(1, 400), _weightsPolicy(128, 1),
      _biasesPolicy(1, 1) {
  auto initRandom = [](Tensor &t) {
    for (int i = 0; i < t.size(); ++i)
      t[i] = 0.001f * (rand() % 100 - 50);
  };

  initRandom(_weights1);
  initRandom(_weights2);
  initRandom(_weightsPolicy);
}

Network::~Network() = default;

/**
 * @brief load a model from file
 */
bool Network::loadModel(const std::string &modelPath) {
  std::ifstream file(modelPath, std::ios::binary);
  if (!file.is_open()) {
    Logger::getInstance().addLog("Failed to open model file: " + modelPath);
    return false;
  }
  auto loadTensor = [&](Tensor &tensor) {
    file.read(reinterpret_cast<char *>(tensor.values.data()),
              tensor.size() * sizeof(float));
  };
  loadTensor(_weights1);
  loadTensor(_biases1);
  loadTensor(_weights2);
  loadTensor(_biases2);
  loadTensor(_weightsPolicy);
  loadTensor(_biasesPolicy);

  if (!file) {
    Logger::getInstance().addLog("Error reading model file: " + modelPath);
    return false;
  }
  if (file.fail()) {
    Logger::getInstance().addLog("Failed to read model data from file: " +
                                 modelPath);
    return false;
  }
  Logger::getInstance().addLog("Successfully loaded model from: " + modelPath);
  return true;
}

/**
 * @brief Mathematic engine
 * @note Fully connected layer implementation
 * @param input Input tensor
 * @param weights Weights tensor
 * @param biases Biases tensor
 * @param output Output tensor
 */
void Network::denseLayer(const Tensor &input, const Tensor &weights,
                         const Tensor &biases, Tensor &output) {
  int inputSize = input.cols;
  int outputSize = weights.cols;

  const float *inputData = input.values.data();
  const float *weightData = weights.values.data();
  const float *biasData = biases.values.data();
  float *outputData = output.values.data();

  for (int neuron = 0; neuron < outputSize; ++neuron) {
    float sum = biasData[neuron];
    int n = 0;
    for (; n <= inputSize - 4; n += 4) {
      sum += inputData[n] * weightData[n * outputSize + neuron];
      sum += inputData[n + 1] * weightData[(n + 1) * outputSize + neuron];
      sum += inputData[n + 2] * weightData[(n + 2) * outputSize + neuron];
      sum += inputData[n + 3] * weightData[(n + 3) * outputSize + neuron];
    }
    for (; n < inputSize; ++n) {
      sum += inputData[n] * weightData[n * outputSize + neuron];
    }
    outputData[neuron] = sum;
  }
}

/**
 * @brief Apply ReLU activation function
 * @param tensor Tensor to apply ReLU on
 * @note apply a function that set to 0 all negative values
 * With ReLU, the output is max(0, x)
 */
void Network::relu(Tensor &tensor) {
  std::replace_if(
      tensor.values.begin(), tensor.values.end(),
      [](float val) { return val < 0.0f; }, 0.0f);
}

/**
 * @brief Apply Softmax activation function
 * @param values Vector of values to apply Softmax on
 * @note Softmax converts a vector of values into a probability distribution.
 * The function are use to have the sum of all output equal to 1.
 */
void Network::softmax(std::vector<float> &values) {
  if (values.empty())
    return;

  float maxVal = *std::max_element(values.begin(), values.end());
  float sumExp = 0.0f;

  std::transform(values.begin(), values.end(), values.begin(),
                 [&sumExp, maxVal](float val) {
                   float expVal = std::exp(val - maxVal);
                   sumExp += expVal;
                   return expVal;
                 });

  if (sumExp > 0.0f) {
    std::transform(values.begin(), values.end(), values.begin(),
                   [sumExp](float val) { return val / sumExp; });
  }
}

/**
 * @brief Predict the output given an input
 * @param board Input board state
 * @return Output struct containing policy and value
 * @note This method performs a forward pass through the network to
 * generate predictions based on the input data.
 */
Output Network::predict(const Board &board) {
  const auto &myBoard = board.getMyBoard();
  const auto &opponentBoard = board.getOpponentBoard();
  float *inputData = _inputBuffer.values.data();

  for (int n = 0; n < 400; ++n) {
    inputData[n] = (float)myBoard.test(n) - (float)opponentBoard.test(n);
  }
  denseLayer(_inputBuffer, _weights1, _biases1, _hiddenBuffer);
  relu(_hiddenBuffer);

  denseLayer(_hiddenBuffer, _weights2, _biases2, _policyLogitsBuffer);
  std::vector<float> policyLogits = _policyLogitsBuffer.values;
  softmax(policyLogits);

  denseLayer(_hiddenBuffer, _weightsPolicy, _biasesPolicy, _valueOutBuffer);
  float value = std::tanh(_valueOutBuffer[0]);
  return {policyLogits, value};
}
