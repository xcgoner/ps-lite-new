#ifndef LRPROX_SOFTMAX_H_
#define LRPROX_SOFTMAX_H_

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
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
    const MatrixXd& getWeight();
//
    void updateWeight(const std::vector<double>& weight);
    void updateWeight(const MatrixXd& weight);

    void outputWeight(std::vector<double>& weight);

    MatrixXi onehot_encoder(const VectorXi &y);
//
//    bool saveModel(const std::string &filename);

//  std::string DebugInfo();

  private:
    void initWeight_();

    MatrixXd logsumexp(const MatrixXd &X);

    // num_dims_ * num_classes_
    MatrixXd weight_;
    int num_dims_;
    int num_classes_;

  };

}  // namespace lrprox

#endif  // LRPROX_SOFTMAX_H_
