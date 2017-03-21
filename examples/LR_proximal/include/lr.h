#ifndef LRPROX_LR_H_
#define LRPROX_LR_H_

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace lrprox {

  class LR {
  public:
    explicit LR(int num_dims);

    ~LR() {};

    double cost(const MatrixXd &X, const VectorXi &y);

    VectorXd grad(const MatrixXd &X, const VectorXi &y);
//
//    VectorXi predict(const MatrixXd &X);
//
    const VectorXd& getWeight();

    void updateWeight(const std::vector<double>& weight);
    void updateWeight(const VectorXd& weight);
//
//    bool saveModel(std::string &filename);

//  std::string DebugInfo();

  private:
    void initWeight_();

    VectorXd weight_;
    int num_dims_;

  };

}  // namespace lrprox

#endif  // LRPROX_LR_H_
