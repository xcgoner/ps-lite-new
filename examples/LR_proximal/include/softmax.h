#ifndef LRPROX_SOFTMAX_H_
#define LRPROX_SOFTMAX_H_

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;

namespace lrprox {

  class SOFTMAX {
  public:
    explicit SOFTMAX(int num_dims, int num_classes);

    ~SOFTMAX() {};

    double cost(const MatrixXd &X, const MatrixXi &Y);

    MatrixXd grad(const MatrixXd &X, const MatrixXi &Y);
////
////    VectorXi predict(const MatrixXd &X);
////
//    const VectorXd& getWeight();
//
//    void updateWeight(const std::vector<double>& weight);
//    void updateWeight(const VectorXd& weight);
//
//    bool saveModel(const std::string &filename);

//  std::string DebugInfo();

  private:
    void initWeight_();

    // num_dims_ * num_classes_
    MatrixXd weight_;
    int num_dims_;
    int num_classes_;

  };

}  // namespace lrprox

#endif  // LRPROX_SOFTMAX_H_
