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


  return 0;
}