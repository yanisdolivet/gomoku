/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Network
*/

#include "Network.hpp"
#include <algorithm>
#include <cmath>

/**
 * @brief Construct a new Network:: Network object
 */
Network::Network() : _inputChannels(0), _numResBlocks(0) {}

/**
 * @brief Destroy the Network:: Network object
 */
Network::~Network() {}

/**
 * @brief Read data from a file into a vector
 *
 * @tparam T Type of data to read
 * @param file Input file stream
 * @param vec Vector to store the read data
 * @param count Number of elements to read
 */
template <typename T>
void readData(std::ifstream &file, std::vector<T> &vec, size_t count) {
  vec.resize(count);
  file.read(reinterpret_cast<char *>(vec.data()), count * sizeof(T));
}

/**
 * @brief Read a tensor from a file
 *
 * @param file Input file stream
 * @param tensor Tensor to store the read data
 * @param n Number of channels
 * @param ch Number of channels
 * @param r Number of rows
 * @param c Number of columns
 */
void readTensor(std::ifstream &file, Tensor &tensor, int n, int ch, int r,
                int c) {
  tensor = Tensor(n, ch, r, c);
  file.read(reinterpret_cast<char *>(tensor.values.data()),
            tensor.size() * sizeof(float));
}

/**
 * @brief Load a model from a file
 *
 * @param modelPath Path to the model file
 * @return true if the model was loaded successfully
 * @return false if the model could not be loaded
 */
bool Network::loadModel(const std::string &modelPath) {
  std::ifstream file(modelPath, std::ios::binary);
  if (!file.is_open()) {
    Logger::addLogGlobal("Error: Cannot open model file: " + modelPath);
    return false;
  }

  unsigned int magic;
  file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (magic != 0x5245534E) { // RESN
    Logger::addLogGlobal(
        "Error: Invalid model file format (Magic Number Mismatch)");
    return false;
  }

  int inputCh, numBlocks;
  file.read(reinterpret_cast<char *>(&inputCh), sizeof(inputCh));
  file.read(reinterpret_cast<char *>(&numBlocks), sizeof(numBlocks));

  _inputChannels = inputCh;
  _numResBlocks = numBlocks;

  readTensor(file, _convInput.weights, 64, inputCh, 3, 3);
  readTensor(file, _convInput.biases, 1, 64, 1, 1);

  readTensor(file, _bnInput.gamma, 1, 64, 1, 1);
  readTensor(file, _bnInput.beta, 1, 64, 1, 1);
  readTensor(file, _bnInput.mean, 1, 64, 1, 1);
  readTensor(file, _bnInput.var, 1, 64, 1, 1);

  _backbone.clear();
  for (int i = 0; i < numBlocks; ++i) {
    ResBlock block;
    readTensor(file, block.conv1.weights, 64, 64, 3, 3);
    readTensor(file, block.conv1.biases, 1, 64, 1, 1);
    readTensor(file, block.bn1.gamma, 1, 64, 1, 1);
    readTensor(file, block.bn1.beta, 1, 64, 1, 1);
    readTensor(file, block.bn1.mean, 1, 64, 1, 1);
    readTensor(file, block.bn1.var, 1, 64, 1, 1);

    readTensor(file, block.conv2.weights, 64, 64, 3, 3);
    readTensor(file, block.conv2.biases, 1, 64, 1, 1);
    readTensor(file, block.bn2.gamma, 1, 64, 1, 1);
    readTensor(file, block.bn2.beta, 1, 64, 1, 1);
    readTensor(file, block.bn2.mean, 1, 64, 1, 1);
    readTensor(file, block.bn2.var, 1, 64, 1, 1);

    _backbone.push_back(block);
  }

  readTensor(file, _policyConv.weights, 1, 64, 1, 1);
  readTensor(file, _policyConv.biases, 1, 1, 1, 1);
  readTensor(file, _policyBN.gamma, 1, 1, 1, 1);
  readTensor(file, _policyBN.beta, 1, 1, 1, 1);
  readTensor(file, _policyBN.mean, 1, 1, 1, 1);
  readTensor(file, _policyBN.var, 1, 1, 1, 1);

  readTensor(file, _valueConv.weights, 1, 64, 1, 1);
  readTensor(file, _valueConv.biases, 1, 1, 1, 1);
  readTensor(file, _valueBN.gamma, 1, 1, 1, 1);
  readTensor(file, _valueBN.beta, 1, 1, 1, 1);
  readTensor(file, _valueBN.mean, 1, 1, 1, 1);
  readTensor(file, _valueBN.var, 1, 1, 1, 1);

  readTensor(file, _valueReduce.weights, 1, 1, 20, 20);
  readTensor(file, _valueReduce.biases, 1, 1, 1, 1);
  _inputTensor = Tensor(1, inputCh, 20, 20);
  _buffer1 = Tensor(1, 64, 20, 20);
  _buffer2 = Tensor(1, 64, 20, 20);
  _buffer3 = Tensor(1, 64, 20, 20);

  _policyConvOut = Tensor(1, 1, 20, 20);
  _policyLogits = Tensor(1, 400);

  _valueConvOut = Tensor(1, 1, 20, 20);
  _valueRedOut = Tensor(1, 1, 1, 1);

  Logger::addLogGlobal("Model loaded successfully. ResBlocks: " +
                       std::to_string(numBlocks));
  return true;
}

/**
 * @brief Predict the next move for a given board state
 *
 * @param board The board state to predict the next move for
 * @return Output The predicted move and its probability
 */
Output Network::predict(const Board &board, int currentPlayer) {
  if (_backbone.empty()) {
    return {{}, 0.0f};
  }

  std::fill(_inputTensor.values.begin(), _inputTensor.values.end(), 0.0f);

  const auto &selfBoard =
      (currentPlayer == 1) ? board.getMyBoard() : board.getOpponentBoard();
  const auto &enemyBoard =
      (currentPlayer == 1) ? board.getOpponentBoard() : board.getMyBoard();

  int lastMoveIdx = board.getLastMoveIndex();

  std::bitset<AREA> myThreats = board.getThreatCandidates(currentPlayer);
  int enemyPlayer = (currentPlayer == 1) ? 2 : 1;
  std::bitset<AREA> opThreats = board.getThreatCandidates(enemyPlayer);

  Tensor &input = _inputTensor;

  for (int y = 0; y < 20; ++y) {
    for (int x = 0; x < 20; ++x) {
      int idx = y * 20 + x;

      if (selfBoard.test(idx)) {
        input.at(0, 0, y, x) = 1.0f;
      }

      if (enemyBoard.test(idx)) {
        input.at(0, 1, y, x) = 1.0f;
      }

      if (idx == lastMoveIdx) {
        input.at(0, 2, y, x) = 1.0f;
      }

      if (myThreats.test(idx)) {
        input.at(0, 3, y, x) = 1.0f;
      }

      if (opThreats.test(idx)) {
        input.at(0, 4, y, x) = 1.0f;
      }
    }
  }

  conv2d(_inputTensor, _convInput, _buffer1, 1, 1);
  batchNorm(_buffer1, _bnInput);
  relu(_buffer1);

  Tensor *current = &_buffer1;
  Tensor *next = &_buffer2;
  Tensor *temp = &_buffer3;

  for (const auto &block : _backbone) {
    conv2d(*current, block.conv1, *next, 1, 1);
    batchNorm(*next, block.bn1);
    relu(*next);
    conv2d(*next, block.conv2, *temp, 1, 1);
    batchNorm(*temp, block.bn2);
    add(*temp, *current);
    relu(*temp);
    Tensor *oldCurrent = current;
    current = temp;
    temp = next;
    next = oldCurrent;
  }
  conv2d(*current, _policyConv, _policyConvOut, 1, 0);
  batchNorm(_policyConvOut, _policyBN);
  relu(_policyConvOut);

  std::copy(_policyConvOut.values.begin(), _policyConvOut.values.end(),
            _policyLogits.values.begin());

  softmax(_policyLogits.values);

  float sumP = 0.0f;
  std::vector<std::pair<float, int>> topMoves;
  for (int i = 0; i < 400; ++i) {
    float p = _policyLogits.values[i];
    sumP += p;
    topMoves.push_back({p, i});
  }
  std::sort(topMoves.rbegin(), topMoves.rend());

  if (topMoves[0].first > 0.1f) {
    std::string logMsg = "Network Top 3: ";
    for (int k = 0; k < 3; ++k) {
      int idx = topMoves[k].second;
      logMsg += "[" + std::to_string(idx % 20) + "," +
                std::to_string(idx / 20) +
                "]:" + std::to_string(topMoves[k].first) + " ";
    }
    Logger::addLogGlobal(logMsg + " (SumP=" + std::to_string(sumP) + ")");
  }

  conv2d(*current, _valueConv, _valueConvOut, 1, 0);
  batchNorm(_valueConvOut, _valueBN);
  relu(_valueConvOut);

  conv2d(_valueConvOut, _valueReduce, _valueRedOut, 1, 0);

  float val = std::tanh(_valueRedOut[0]);

  return {_policyLogits.values, val};
}

/**
 * @brief Perform a 2D convolution operation
 * @param input Input tensor
 * @param layer Convolution layer
 * @param output Output tensor
 * @param stride Stride of the convolution
 * @param padding Padding of the convolution
 */
void Network::conv2d(const Tensor &input, const ConvLayer &layer,
                     Tensor &output, int stride, int padding) {
  int Cin = input.channels;
  int Hin = input.rows;
  int Win = input.cols;

  int Cout = layer.weights.num;
  int K = layer.weights.rows;

  int Hout = (Hin + 2 * padding - K) / stride + 1;
  int Wout = (Win + 2 * padding - K) / stride + 1;

  std::fill(output.values.begin(), output.values.end(), 0.0f);

  const float *inPtr = input.values.data();
  const float *wPtr = layer.weights.values.data();
  const float *bPtr = layer.biases.values.data();
  float *outPtr = output.values.data();

  for (int oc = 0; oc < Cout; ++oc) {
    float bias = bPtr[oc];

    for (int oh = 0; oh < Hout; ++oh) {
      for (int ow = 0; ow < Wout; ++ow) {

        float sum = bias;

        int h_start = oh * stride - padding;
        int w_start = ow * stride - padding;

        const float *w_oc_ptr = wPtr + oc * (Cin * K * K);

        for (int ic = 0; ic < Cin; ++ic) {
          const float *in_ic_ptr = inPtr + ic * (Hin * Win);
          const float *w_ic_ptr = w_oc_ptr + ic * (K * K);

          for (int kh = 0; kh < K; ++kh) {
            int ih = h_start + kh;
            if (ih >= 0 && ih < Hin) {
              const float *in_row_ptr = in_ic_ptr + ih * Win;
              const float *w_row_ptr = w_ic_ptr + kh * K;

              for (int kw = 0; kw < K; ++kw) {
                int iw = w_start + kw;
                if (iw >= 0 && iw < Win) {
                  sum += in_row_ptr[iw] * w_row_ptr[kw];
                }
              }
            }
          }
        }
        outPtr[oc * (Hout * Wout) + oh * Wout + ow] = sum;
      }
    }
  }
}

/**
 * @brief Perform batch normalization
 *
 * @param input Input tensor
 * @param layer Batch normalization layer
 */
void Network::batchNorm(Tensor &input, const BNLayer &layer) {
  int C = input.channels;
  int HW = input.rows * input.cols;
  float *data = input.values.data();

  const float *gamma = layer.gamma.values.data();
  const float *beta = layer.beta.values.data();
  const float *mean = layer.mean.values.data();
  const float *var = layer.var.values.data();

  float epsilon = 1e-5f;

  for (int c = 0; c < C; ++c) {
    float scale = gamma[c] / std::sqrt(var[c] + epsilon);
    float shift = beta[c] - (mean[c] * scale);

    for (int i = 0; i < HW; ++i) {
      data[c * HW + i] = data[c * HW + i] * scale + shift;
    }
  }
}

/**
 * @brief Apply ReLU activation
 *
 * @param tensor Input tensor
 */
void Network::relu(Tensor &tensor) {
  std::replace_if(
      tensor.values.begin(), tensor.values.end(),
      [](float v) { return v < 0.0f; }, 0.0f);
}

/**
 * @brief Element-wise addition of two tensors
 *
 * @param input Input tensor
 * @param other Other tensor
 */
void Network::add(Tensor &input, const Tensor &other) {
  for (size_t i = 0; i < input.values.size(); ++i) {
    input.values[i] += other.values[i];
  }
}

/**
 * @brief Apply softmax activation
 *
 * @param values Input values
 */
void Network::softmax(std::vector<float> &values) {
  float maxVal = *std::max_element(values.begin(), values.end());

  float sum = 0.0f;
  for (float &v : values) {
    v = std::exp(v - maxVal);
    sum += v;
  }
  std::transform(values.begin(), values.end(), values.begin(),
                 [sum](float v) { return v / sum; });
}
