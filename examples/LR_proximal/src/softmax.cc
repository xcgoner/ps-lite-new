#include "cmath"
#include "softmax.h"
#include <ctime>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXi;
using Eigen::ArrayXXd;

namespace lrprox {

  SOFTMAX::SOFTMAX(int num_dims, int num_classes)
    : num_dims_(num_dims), num_classes_(num_classes) {
    initWeight_();
  }

  double SOFTMAX::cost(const MatrixXd &X, const MatrixXi &Y) {
    // X: n * num_dims, y: n * num_classes
    // minimize negative log prob
    ArrayXXd term1 = (X * weight_).array();
    // do not take average
    return ((logsumexp(term1.matrix()).replicate(1, term1.cols()).array() - term1) * Y.cast<double>().array()).sum();
  }

  MatrixXd SOFTMAX::grad(const MatrixXd &X, const MatrixXi &Y) {
    MatrixXd term1 = (X * weight_).array().exp().matrix();
    ArrayXXd P = term1.array() / term1.rowwise().sum().replicate(1, term1.cols()).array();

    return X.transpose() * (P - Y.cast<double>().array()).matrix();
//    return weight_;
  }
//
//  VectorXi predict(const MatrixXd &X);
//
  const MatrixXd& SOFTMAX::getWeight() {
    return weight_;
  }
  void SOFTMAX::updateWeight(const std::vector<double>& weight) {
    for (int i = 0; i < weight_.cols(); i++) {
      weight_.col(i) = VectorXd::Map(&weight[i*weight_.rows()], weight_.rows());
    }
  }
  void SOFTMAX::updateWeight(const MatrixXd& weight) {
    weight_ = weight;
  }

  void SOFTMAX::outputWeight(std::vector<double>& weight) {
    weight.resize(weight_.rows() * weight_.cols());
    for (int i = 0; i < weight_.cols(); i++) {
       VectorXd::Map(&weight[i*weight_.rows()], weight_.rows()) = weight_.col(i);
    }
  }

  MatrixXi SOFTMAX::onehot_encoder(const VectorXi &y) {
    std::vector<int> labels;
    std::map<int, int> label_to_idx;
    for (int i = 0; i < y.size(); i++) {
      if (label_to_idx.count(y(i)) == 0) {
        label_to_idx[y(i)] = 1;
        labels.push_back(y(i));
      }
    }
    std::sort(labels.begin(), labels.end());
    for (int i = 0; i < labels.size(); i++) {
      label_to_idx[labels[i]] = i;
    }
    MatrixXi Y = MatrixXi::Zero(y.size(), labels.size());
    for (int i = 0; i < y.size(); i++) {
      Y(i, label_to_idx[y(i)]) = 1;
    }
    return Y;
  }

//
//  bool saveModel(std::string &filename);

  void SOFTMAX::initWeight_() {
//    weight_ = MatrixXd::Random(num_dims_, num_classes_);
    weight_ = MatrixXd::Ones(num_dims_, num_classes_);
  }

  MatrixXd SOFTMAX::logsumexp(const MatrixXd &X) {
    MatrixXd max_X = X.rowwise().maxCoeff();
    return max_X + (X - max_X.replicate(1, X.cols())).array().exp().matrix().rowwise().sum().array().log().matrix();
  }


//bool LR::SaveModel(std::string& filename) {
//  std::ofstream fout(filename.c_str());
//  fout << num_feature_dim_ << std::endl;
//  for (int i = 0; i < num_feature_dim_; ++i) {
//    fout << weight_[i] << ' ';
//  }
//  fout << std::endl;
//  fout.close();
//  return true;
//}

//std::string LR::DebugInfo() {
//  std::ostringstream out;
//  for (size_t i = 0; i < weight_.size(); ++i) {
//    out << weight_[i] << " ";
//  }
//  return out.str();
//}

//  float LR::Sigmoid_(std::vector<float> feature) {
//    float z = 0;
//    for (size_t j = 0; j < weight_.size(); ++j) {
//      z += weight_[j] * feature[j];
//    }
//    return 1. / (1. + exp(-z));
//  }

} // namespace lrprox
