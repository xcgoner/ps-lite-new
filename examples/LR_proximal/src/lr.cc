#include "ps/ps.h"
#include "cmath"
#include "lr.h"
#include <ctime>
#include <iostream>
#include <vector>
#include <fstream>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;

namespace lrprox {

  LR::LR(int num_dims)
    : num_dims_(num_dims) {
    initWeight_();
  }

  double LR::cost(const MatrixXd &X, const VectorXi &y) {
    // minimize negative log prob
    VectorXd term_in_sigmoid = - y.cast<double>().cwiseProduct(X * weight_);
    // do not take average
    return term_in_sigmoid.array().exp().log1p().matrix().sum();
  }

  VectorXd LR::grad(const MatrixXd &X, const VectorXi &y) {
    ArrayXd term1 = 1 - ((- y.cast<double>().cwiseProduct(X * weight_)).array().exp() + 1).inverse();
    return -X.transpose() * (term1 * y.cast<double>().array()).matrix();
  }
//
//  VectorXi predict(const MatrixXd &X);
//
  const VectorXd& LR::getWeight() {
    return weight_;
  }
  void LR::updateWeight(const std::vector<double>& weight) {
    for (int i = 0; i < weight.size(); i++) {
      weight_(i) = weight[i];
    }
  }
//
//  bool saveModel(std::string &filename);

  void LR::initWeight_() {
//    weight_ = VectorXd::Random(num_dims_);
    weight_ = VectorXd::Ones(num_dims_);
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
