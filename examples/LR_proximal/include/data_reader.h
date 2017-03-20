#ifndef DATA_READER_H_
#define DATA_READER_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

#include "util.h"

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;

namespace lrprox {

  class data_reader {
  public:
    // read sparse classification data, libsvm style
    explicit data_reader(const std::string &filename, int num_feature_dim) {
      std::ifstream input;
      std::string line, buf;

      // count lines
      input.open(filename);
      int line_counter = 0;
      while (std::getline(input, line)) {
        line_counter++;
      }
      input.close();

      // initialize
      // add one column
      X = MatrixXd::Zero(line_counter, num_feature_dim + 1);
      y = VectorXi::Zero(line_counter);

      input.open(filename);
      int line_idx = 0;
      while (std::getline(input, line)) {
        std::istringstream in(line);
        in >> buf;
        y(line_idx) = util::ToInt(buf);
        while (in >> buf) {
          auto ss = util::Split(buf, ':');
          // read sparse data
          X(line_idx, util::ToInt(ss[0]) - 1) = util::ToDouble(ss[1]);
        }
        X(line_idx, num_feature_dim) = 1;
        line_idx++;
      }
      input.close();
    }

    virtual ~data_reader() {
    }

    const MatrixXd &getX() {
      return X;
    }

    const VectorXi &gety() {
      return y;
    }


  private:
    MatrixXd X;
    VectorXi y;

  };

}  // namespace lrprox

#endif  // DATA_READER_H_