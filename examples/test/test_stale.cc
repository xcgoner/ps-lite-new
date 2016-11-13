#include "ps/ps.h"
#include <iostream>

using namespace std;
using namespace ps;

typedef float ValueType;

void StartServer() {
    if (!IsServer()) return;
    auto server = new KVServer<ValueType>(0);
    server->set_request_handle(KVServerDefaultHandle<ValueType>());
    RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
    if (!IsWorker()) return;
    KVWorker<ValueType> kv(0);

    // init
    int num = 4;
    std::vector<Key> keys(num);
    std::vector<ValueType> vals(num);

    int rank = MyRank();
    srand(rank + 7);

    std::vector<Key> keys_p(num);
    std::vector<ValueType> vals_p(num);
    for (int i = 0; i < num; ++i) {
        keys_p[i] = i;
    }

    // push
    if (rank % 2 == 0) {
        for (int i = 0; i < num; ++i) {
            keys[i] = i;
            vals[i] = i;
        }
        std::this_thread::sleep_for(chrono::seconds(3));
        int ts = kv.Push(keys, vals);
        cout << rank << ": " << ts << endl;
        kv.Wait(ts);
        cout << "Pushed" << endl;
    }
    else {
        for (int i = 0; i < num; ++i) {
            keys[i] = i;
            vals[i] = 0;
        }
        int ts = kv.Push(keys, vals);
        cout << rank << ": " << ts << endl;
        kv.Wait(ts);
        cout << "Pushed" << endl;
    }
    kv.Wait(kv.Pull(keys_p, &vals_p));
    for (int i = 0; i < num; ++i) {
        cout << vals_p[i] << ", ";
    }
    cout << endl;

}

int main(int argc, char *argv[]) {

    StartServer();

    ps::Start();
    // do nothing
    if (ps::IsWorker()) {
        RunWorker();
    }
    ps::Finalize();
    return 0;
}