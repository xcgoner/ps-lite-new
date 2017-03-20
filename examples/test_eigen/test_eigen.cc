#include "ps/ps.h"
#include <iostream>
#include <Eigen/Dense>
#include <pthread.h>
#include <unistd.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
using namespace std;
//using Eigen::MatrixXd;
//using Eigen::VectorXd;

void *someThread(void *ptr) {
  while(1) {
    sleep(1);
    cout << "the thread is running" << endl;
  }
}

int main(int argc, char *argv[]) {
    ps::Start();
    // do nothing
    cout << "Launched" << endl;
    if (ps::IsServer()) {
        cout << "A Server!" << endl;
    }
    if (ps::IsWorker()) {
        cout << "A Worker!" << endl;
//      MatrixXd m(2,2);
//      m(0,0) = 3;
//      m(1,0) = 2.5;
//      m(0,1) = -1;
//      m(1,1) = m(1,0) + m(0,1);
//
//      VectorXd v(2);
//      v << 1 , 2;
//
//      std::cout << m * v << std::endl;
      pthread_t thread1;
      pthread_create(&thread1, NULL, someThread, NULL);
      pthread_detach(thread1);
      int i = 0;
      while(1) {
        sleep(1);
        cout << "the worker is running, i = " << i << endl;
        i++;
        if (i==7) {
          // test cancelling
          pthread_cancel(thread1);
        }
      }


    }
    if (ps::IsScheduler()) {
        cout << "A Scheduler!" << endl;
    }

    ps::Finalize();
    return 0;
}

#pragma clang diagnostic pop