//#include "ps/ps.h"
#include <iostream>
#include <vector>
#include "lr.h"

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;

using namespace std;
//using namespace ps;

int main() {
  lrprox::LR lr = lrprox::LR(4);
  MatrixXd X(3, 4);
  VectorXi y(3);
  vector<double> a = {1, -1, 1};
  X << 1,2,3,4,5,6,7,8,9,10,11,12;
  X = X / 10;
  for (int i = 0; i < a.size(); i++) {
    y(i) = a[i];
  }
  cout << X << endl;
  cout << y << endl;
  cout << "---------" << endl;
  cout << lr.cost(X, y) << endl;
  cout << "---------" << endl;
  cout << lr.grad(X, y) << endl;

//  test results should be
//    0.1 0.2 0.3 0.4
//    0.5 0.6 0.7 0.8
//    0.9   1 1.1 1.2
//    1
//    -1
//    1
//        ---------
//    2.99979
//        ---------
//    0.42524
//    0.489955
//    0.554669
//    0.619384

  return 0;
}