//#include "ps/ps.h"
#include <iostream>
#include <vector>
#include "lr.h"
#include "prox_l1.h"
#include "prox_l2.h"

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

  lrprox::prox_l1 prox_op = lrprox::prox_l1(0.1);
  cout << "---------" << endl;
  cout << prox_op.cost(lr.grad(X, y)) << endl;
  cout << "---------" << endl;
  cout << prox_op.proximal(lr.grad(X, y), 0.3) << endl;

  lrprox::prox_l2 prox_op2 = lrprox::prox_l2(0.1);
  cout << "---------" << endl;
  cout << prox_op2.cost(lr.grad(X, y)) << endl;
  cout << "---------" << endl;
  cout << prox_op2.proximal(lr.grad(X, y), 0.3) << endl;


//  test results should be
//  0.1 0.2 0.3 0.4
//  0.5 0.6 0.7 0.8
//  0.9   1 1.1 1.2
//  1
//  -1
//  1
//      ---------
//  2.99979
//      ---------
//  0.42524
//  0.489955
//  0.554669
//  0.619384
//      ---------
//  0.208925
//      ---------
//  0.39524
//  0.459955
//  0.524669
//  0.589384
//      ---------
//  0.10546
//      ---------
//  0.413143
//  0.476017
//  0.538891
//  0.601764

  return 0;
}