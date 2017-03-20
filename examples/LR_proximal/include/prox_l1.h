//
// Created by cx2 on 3/19/17.
//

#ifndef PROX_L1_H
#define PROX_L1_H

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace lrprox {

  class prox_l1 {
  public:
    explicit prox_l1(double lambda)
    : lambda_(lambda){

    }

    ~prox_l1() {};

    double cost(const VectorXd& v) {
      return lambda_ * v.lpNorm<1>();
    }

    // gamma is the step size
    VectorXd proximal(const VectorXd& v, double gamma) {
      return ((v.array() - lambda_ * gamma).max(0) - (-v.array() - lambda_ * gamma).max(0)).matrix();
    }

  private:

    double lambda_;

  };

}  // namespace lrprox

#endif //PROX_L1_H
