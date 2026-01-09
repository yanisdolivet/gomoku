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
    : _inputBuffer(1, 800), _hidden1Buffer(1, 512), _hidden2Buffer(1, 256),
      _policyLogitsBuffer(1, 400), _valueOutBuffer(1, 1),
      _weightsShared1(800, 512), _biasesShared1(1, 512),
      _weightsShared2(512, 256), _biasesShared2(1, 256),
      _weightsPolicy(256, 400), _biasesPolicy(1, 400),
      _weightsValue(256, 1), _biasesValue(1, 1) {
  auto initRandom = [](Tensor &t) {
    for (int i = 0; i < t.size(); ++i)
      t[i] = 0.001f * (rand() % 100 - 50);
  };

  initRandom(_weightsShared1);
  initRandom(_weightsShared2);
  initRandom(_weightsPolicy);
  initRandom(_weightsValue);
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

  loadTensor(_weightsShared1);
  loadTensor(_biasesShared1);
  loadTensor(_weightsShared2);
  loadTensor(_biasesShared2);
  loadTensor(_weightsPolicy);
  loadTensor(_biasesPolicy);
  loadTensor(_weightsValue);
  loadTensor(_biasesValue);

  if (!file) {
    Logger::getInstance().addLog("Error reading model file: " + modelPath);
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
 * The function is used to ensure that the sum of all outputs equals 1.
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
    bool isMe = myBoard.test(n);
    bool isOp = opponentBoard.test(n);

    inputData[n]       = isMe ? 1.0f : 0.0f; // First Channel
    inputData[n + 400] = isOp ? 1.0f : 0.0f; // Second Channel
  }

  denseLayer(_inputBuffer, _weightsShared1, _biasesShared1, _hidden1Buffer);
  relu(_hidden1Buffer);

  denseLayer(_hidden1Buffer, _weightsShared2, _biasesShared2, _hidden2Buffer);
  relu(_hidden2Buffer);

  denseLayer(_hidden2Buffer, _weightsPolicy, _biasesPolicy, _policyLogitsBuffer);
  std::vector<float> policyLogits = _policyLogitsBuffer.values;
  softmax(policyLogits);

  denseLayer(_hidden2Buffer, _weightsValue, _biasesValue, _valueOutBuffer);
  float value = std::tanh(_valueOutBuffer[0]);

  return {policyLogits, value};
}
