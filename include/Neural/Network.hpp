/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Network
*/

#pragma once

#include "Board.hpp"
#include "Logger.hpp"
#include "Tensor.hpp"

#include <fstream>
#include <vector>

struct Output {
  std::vector<float> policy;
  float value;
};

struct ConvLayer {
  Tensor weights;
  Tensor biases;
};

struct BNLayer {
  Tensor gamma;
  Tensor beta;
  Tensor mean;
  Tensor var;
};

struct ResBlock {
  ConvLayer conv1;
  BNLayer bn1;
  ConvLayer conv2;
  BNLayer bn2;
};

class Network {
public:
  Network();
  ~Network();

  bool loadModel(const std::string &modelPath);
  Output predict(const Board &board, int currentPlayer = 1);

private:
  int _inputChannels;
  int _numResBlocks;

  ConvLayer _convInput;
  BNLayer _bnInput;
  std::vector<ResBlock> _backbone;

  ConvLayer _policyConv;
  BNLayer _policyBN;

  ConvLayer _valueConv;
  BNLayer _valueBN;
  ConvLayer _valueReduce;

  Tensor _inputTensor;
  Tensor _buffer1;
  Tensor _buffer2;
  Tensor _buffer3;
  Tensor _policyConvOut;
  Tensor _policyFlat;
  Tensor _policyLogits;

  Tensor _valueConvOut;
  Tensor _valueFlat;
  Tensor _valueHidden;
  Tensor _valueRedOut;

  void conv2d(const Tensor &input, const ConvLayer &layer, Tensor &output,
              int stride = 1, int padding = 0);
  void batchNorm(Tensor &input, const BNLayer &layer);
  void add(Tensor &input, const Tensor &other);
  void relu(Tensor &tensor);
  void softmax(std::vector<float> &values);
};