#include "ps/ps.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char *argv[]) {
    ps::Start();
    // do nothing
    cout << "Launched" << endl;
    if (ps::IsServer()) {
        cout << "A Server!" << endl;
    }
    if (ps::IsWorker()) {
        cout << "A Worker!" << endl;
      MatrixXd m(2,2);
      m(0,0) = 3;
      m(1,0) = 2.5;
      m(0,1) = -1;
      m(1,1) = m(1,0) + m(0,1);

      VectorXd v(2);
      v << 1 , 2;

      std::cout << m * v << std::endl;
    }
    if (ps::IsScheduler()) {
        cout << "A Scheduler!" << endl;
    }

    ps::Finalize();
    return 0;
}
