#include "util.h"
#include <iostream>

using namespace std;

int main() {

  cout << util::ToInt("-123") << endl;
  cout << util::ToDouble("-123.123") << endl;

  return 0;
}