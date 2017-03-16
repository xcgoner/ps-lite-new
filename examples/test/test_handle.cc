#include "ps/ps.h"
#include <iostream>
#include "ps/ssp_push_val.h"

using namespace std;
using namespace ps;

typedef float ValueType;
typedef SSPPushVal<ValueType> PushValueType;

template <typename Val>
struct KVServerSSPHandle {
    void operator()(
            const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
        size_t n = req_data.keys.size();
        KVPairs<Val> res;
        if (req_meta.push) {
            CHECK_EQ(n, req_data.vals.size());
            cout << "Push req, sender: " << req_meta.sender << ", ts:" << req_meta.timestamp << endl;
        } else {
            res.keys = req_data.keys; res.vals.resize(n);
            cout << "Pull req, sender: " << req_meta.sender << ", ts:" << req_meta.timestamp << endl;
        }
        for (size_t i = 0; i < n; ++i) {
            Key key = req_data.keys[i];
            if (req_meta.push) {
                store[key] += req_data.vals[i];
            } else {
                res.vals[i] = store[key];
            }
        }
        server->Response(req_meta, res);
    }
    std::unordered_map<Key, Val> store;
};

void StartServer() {
    if (!IsServer()) return;
    auto server = new KVServer<PushValueType>(0);
    server->set_request_handle(KVServerSSPHandle<PushValueType>());
    RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
    if (!IsWorker()) return;
    KVWorker<PushValueType> kv(0);

    // init
    int num = 4;
    std::vector<Key> keys(num);
    std::vector<PushValueType> vals(num);

    int rank = MyRank();
    srand(rank + 7);

    std::vector<Key> keys_p(num);
    std::vector<PushValueType> vals_p(num);
    for (int i = 0; i < num; ++i) {
        keys_p[i] = i;
    }

    // push
    if (rank % 2 == 0) {
        for (int i = 0; i < num; ++i) {
            keys[i] = i;
            vals[i].UpdateVal(i);
        }
        int ts = kv.Push(keys, vals);
        cout << "push rank: " << rank << ",\tts: " << ts << endl;
        kv.Wait(ts);
//        cout << "Pushed" << endl;
    }
    else {
        for (int i = 0; i < num; ++i) {
            keys[i] = i;
            vals[i].UpdateVal(0);
        }
        int ts = kv.Push(keys, vals);
        cout << "push rank: " << rank << ",\tts: " << ts << endl;
        kv.Wait(ts);
//        cout << "Pushed" << endl;
    }
    int ts = kv.Push(keys, vals);
    cout << "push rank: " << rank << ",\tts: " << ts << endl;
    ts = kv.Pull(keys_p, &vals_p);
    cout << "pull rank: " << rank << ",\tts: " << ts << endl;
    kv.Wait(ts);
    for (int i = 0; i < num; ++i) {
        cout << vals_p[i].nitr_ << ", ";
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