#include "ps/ps.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    ps::Start();
    // do nothing
    cout << "Launched" << endl;
    if (ps::IsServer()) {
        cout << "A Server!" << endl;
    }
    if (ps::IsWorker()) {
        cout << "A Worker!" << endl;
    }
    ps::Finalize();
    return 0;
}
