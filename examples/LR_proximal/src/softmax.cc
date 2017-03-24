#include "cmath"
#include "softmax.h"
#include <ctime>
#include <iostream>
#include <vector>
#include <fstream>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::ArrayXd;

namespace lrprox {

  SOFTMAX::SOFTMAX(int num_dims, int num_classes)
    : num_dims_(num_dims), num_classes_(num_classes) {
    initWeight_();
  }

  double SOFTMAX::cost(const MatrixXd &X, const MatrixXi &Y) {
    // X: n * num_dims, y: n * num_classes
    // minimize negative log prob
    ArrayXd term1 = (X * weight_).array();
    // do not take average


    return term1.exp().matrix().colwise().sum().array().log().sum() - (term1 * Y.cast<double>().array()).sum();
  }

  MatrixXd SOFTMAX::grad(const MatrixXd &X, const MatrixXi &Y) {
    MatrixXd term1 = (X * weight_).array().exp().matrix();
    ArrayXd P = term1.array() / term1.rowwise().sum().replicate(1, term1.cols()).array();

    return X.transpose() * (P - Y.cast<double>().array()).matrix();
  }
////
////  VectorXi predict(const MatrixXd &X);
////
//  const VectorXd& SOFTMAX::getWeight() {
//    return weight_;
//  }
//  void SOFTMAX::updateWeight(const std::vector<double>& weight) {
//    for (int i = 0; i < weight.size(); i++) {
//      weight_(i) = weight[i];
//    }
//  }
//  void SOFTMAX::updateWeight(const VectorXd& weight) {
//    weight_ = weight;
//  }
//
//  bool saveModel(std::string &filename);

  void SOFTMAX::initWeight_() {
//    weight_ = MatrixXd::Random(num_dims_, num_classes_);
    weight_ = MatrixXd::Ones(num_dims_, num_classes_);
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
