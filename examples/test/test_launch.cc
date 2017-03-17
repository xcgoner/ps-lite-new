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
    if (ps::IsScheduler()) {
        cout << "A Scheduler!" << endl;
    }

    // test environment variable
    if (ps::Environment::Get()->find("SEMI_SYNC_MODE") != nullptr && !strcmp(ps::Environment::Get()->find("SEMI_SYNC_MODE"), "1")) {
        cout << "semi-synchronous !" << endl;
    }
    else {
        cout << "something is wrong!" << endl;
    }

    ps::Finalize();
    return 0;
}
