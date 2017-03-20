//
// Created by cx2 on 3/19/17.
//

#ifndef PROX_L2_H
#define PROX_L2_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace lrprox {

  class prox_l2 {
  public:
    explicit prox_l2(double lambda)
        : lambda_(lambda){

    }

    ~prox_l2() {};

    double cost(const VectorXd& v) {
      return lambda_ * v.lpNorm<2>();
    }

    // gamma is the step size
    VectorXd proximal(const VectorXd& v, double gamma) {
      return fmax(0, (1 - lambda_ * gamma / v.lpNorm<2>())) * v;
    }

  private:

    double lambda_;

  };

}  // namespace lrprox

#endif //PROX_L2_H