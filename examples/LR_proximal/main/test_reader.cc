#include "util.h"
#include "data_reader.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {

  cout << "reading file: " << argv[1] << endl;

  lrprox::data_reader dr = lrprox::data_reader(argv[1], 123);

  for (int i = 0; i < 3; i++) {
    cout << dr.getX().row(i) << endl;
    cout << dr.gety()(i) << endl;
  }

  return 0;
}