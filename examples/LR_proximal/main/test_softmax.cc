//#include "ps/ps.h"
#include <iostream>
#include <vector>
#include "softmax.h"
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

  // test data copy
  vector<double> vecx(X.rows()*X.cols());
  for (int i = 0; i < X.cols(); i++) {
    VectorXd::Map(&vecx[i*X.rows()], X.rows()) = X.col(i);
  }
  cout << "---------" << endl;
  for (int i = 0; i < vecx.size(); i++) {
    cout << vecx[i] << " ";
  }
  cout << endl;

  // test data copy back
  cout << "---------" << endl;
  for (int i = 0; i < vecx.size(); i++) {
    vecx[i] = vecx[i] * 2;
  }
  for (int i = 0; i < X.cols(); i++) {
    X.col(i) = VectorXd::Map(&vecx[i*X.rows()], X.rows());
  }
  cout << X << endl;

  // test replicate
  cout << "---------" << endl;
  cout << X.rowwise().sum() << endl;
  cout << X.rowwise().sum().replicate(1, X.cols()) << endl;

  // test transpose
  cout << "---------" << endl;
  cout << X.transpose() << endl;
  cout << "---------" << endl;
  cout << X << endl;

  // test softmax
  lrprox::SOFTMAX softmax = lrprox::SOFTMAX(4, 2);
  // one-hot
  MatrixXi y_onehot = softmax.onehot_encoder(y);
//  y_onehot << 0, 1, 0, 1, 0, 1;
//  y_onehot.transposeInPlace();
  cout << "---------" << endl;
  X << 1,2,3,4,5,6,7,8,9,10,11,12;
  X = X / 10;
  cout << X << endl;
  cout << y_onehot << endl;
  for (int i = 1; i <= 3; i++) {
    cout << "--" << endl;
    MatrixXd w(4, 2);
    w.col(0) << 1, 2, 3, 4;
    w.col(1) << 5, 6, 7, 8;
    w = (-w.eval().array()/10 *i).matrix();
    softmax.updateWeight(w);
    cout << softmax.getWeight() << endl;
    cout << softmax.cost(X, y_onehot) << endl;
    cout << softmax.grad(X, y_onehot) << endl;
  }

  // test proximal
  lrprox::prox_l1 prox_op1 = lrprox::prox_l1(0.1);
  lrprox::prox_l2 prox_op2 = lrprox::prox_l2(0.1);
  double gamma = 0.3;
  cout << "---------" << endl;
  VectorXd x(Eigen::Map<VectorXd>(X.data(), X.cols()*X.rows()));
  cout << prox_op1.cost(X) << endl;
  cout << prox_op1.cost(x) << endl;
  cout << prox_op1.proximal(X, gamma) << endl;
  cout << prox_op1.proximal(x, gamma) << endl;
  cout << prox_op2.cost(X) << endl;
  cout << prox_op2.cost(x) << endl;
  cout << prox_op2.proximal(X, gamma) << endl;
  cout << prox_op2.proximal(x, gamma) << endl;

  // test outputWeight
  vector<double> w_vec;
  cout << "---------" << endl;
  softmax.outputWeight(*(&w_vec));
  cout << softmax.getWeight() << endl;
  for (int i = 0; i < w_vec.size(); i++) {
    cout << w_vec[i] << " ";
  }
  cout << endl;
  w_vec[2] = 100;
  softmax.updateWeight(w_vec);
  cout << softmax.getWeight() << endl;
  for (int i = 0; i < w_vec.size(); i++) {
    cout << w_vec[i] << " ";
  }
  cout << endl;
  MatrixXd xtmp = softmax.getWeight();
  xtmp.data()[0] = -100;
  for (int i = 0; i < w_vec.size(); i++) {
    cout << xtmp.data()[i] << " ";
  }
  cout << endl;

  return 0;
}