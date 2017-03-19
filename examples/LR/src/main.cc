#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include "ps/ps.h"

#include "lr.h"
#include "util.h"
#include "data_iter.h"

const int kSyncMode = -1;

template <typename Val>
class KVStoreDistServer {
public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    ps_server_->set_request_handle(
      std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));

    // read environment variables
    sync_mode_ = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");
    learning_rate_ = distlr::ToFloat(ps::Environment::Get()->find("LEARNING_RATE"));
    nsamples_ = distlr::ToInt(ps::Environment::Get()->find("NSAMPLES"));
    ndims_ = distlr::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));
    // TODO: regularization

    std::string mode = sync_mode_ ? "sync" : "async";
    std::cout << "Server mode: " << mode << std::endl;

    // initialize timestamp
    global_ts_ = 0;

  }

  ~KVStoreDistServer() {
    if (ps_server_) {
      delete ps_server_;
    }
  }

private:

  // threadsafe
  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<Val>& req_data,
                  ps::KVServer<Val>* server) {
    int key = DecodeKey(req_data.keys[0]);
    auto& weights = weights_[key];

//    size_t n = req_data.keys.size();
    size_t n = ndims_;
    if (req_meta.push) {
      if (weights.empty()) {
        CHECK_EQ(n, req_data.vals.size());
        std::cout << "Init weight" << std::endl;
        weights.resize(n);
        for (int i = 0; i < n; ++i) {
          weights[i] = req_data.vals[i];
        }
        server->Response(req_meta);
      } else {
          CHECK_EQ(n+1, req_data.vals.size());
          if (sync_mode_) {
            auto &merged = merge_buf_[key];
            // initialization
            if (merged.vals.empty()) {
              merged.vals.resize(n, 0);
              merged.naggregates = 0;
            }

            for (int i = 0; i < n; ++i) {
              merged.vals[i] += req_data.vals[i];
            }
            merged.naggregates += req_data.vals[n];
//            if (req_data.vals[n]!=100) {
//              std::cout << "batchsize: " << req_data.vals[n] << std::endl;
//              std::cout << "total size: " << merged.naggregates  << std::endl;
//            }

            server->Response(req_meta);
            if (merged.naggregates == nsamples_) {
//              std::cout << "apply gradients" << std::endl;
              // update the weight
              for (size_t i = 0; i < n; ++i) {
                // gradient descent
                weights[i] -= learning_rate_ * merged.vals[i] / merged.naggregates;
                // TODO: proximal step
              }
              // timestamp
              global_ts_++;
              merged.vals.clear();
              merged.naggregates = 0;
            }
          } else { // async push
            for (size_t i = 0; i < n; ++i) {
              // SGD
              weights[i] -= learning_rate_ * req_data.vals[i] / req_data.vals[n];
              // TODO: proximal step
            }
            server->Response(req_meta);
          }
      }
    } else { // pull
      CHECK(!weights_.empty()) << "init " << key << " first";

      ps::KVPairs<Val> response;
      response.keys = req_data.keys;
      response.vals.resize(n);
      for (size_t i = 0; i < n; ++i) {
        response.vals[i] = weights[i];
      }
      // timestamp
      response.ts1 = global_ts_;
      response.ts2 = global_ts_ + 1;
      server->Response(req_meta, response);
      // TODO: semi-synchronous
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  bool sync_mode_;
  float learning_rate_;
  // timestamp for iterations
  int global_ts_;
  int nsamples_;
  int ndims_;

  struct MergeBuf {
    std::vector<Val> vals;
    // number of aggregates
    int naggregates;
  };

  std::unordered_map<int, std::vector<Val>> weights_;
  std::unordered_map<int, MergeBuf> merge_buf_;
  ps::KVServer<float>* ps_server_;
};

void StartServer() {
  if (!ps::IsServer()) {
    return;
  }
  auto server = new KVStoreDistServer<float>();
  ps::RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!ps::IsWorker()) {
    return;
  }

  bool sync_mode = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");

  std::string root = ps::Environment::Get()->find("DATA_DIR");
  int num_feature_dim =
    distlr::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));

  int rank = ps::MyRank();
  ps::KVWorker<float>* kv = new ps::KVWorker<float>(0);
  distlr::LR lr = distlr::LR(num_feature_dim);
  lr.SetKVWorker(kv);

  if (rank == 0) {
    auto vals = lr.GetWeight();
    std::vector<ps::Key> keys(vals.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = i;
    }
    kv->Wait(kv->Push(keys, vals));
  }
  ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);

  std::cout << "Worker[" << rank << "]: start working..." << std::endl;
  int num_iteration = distlr::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
  int batch_size = distlr::ToInt(ps::Environment::Get()->find("BATCH_SIZE"));
  int test_interval = distlr::ToInt(ps::Environment::Get()->find("TEST_INTERVAL"));

  // add timer
  std::chrono::time_point<std::chrono::system_clock> start;
  if (rank == 0) {
    start = std::chrono::system_clock::now();
  }

  for (int i = 0; i < num_iteration; ++i) {
    std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
    distlr::DataIter iter(filename, num_feature_dim);
    lr.Train(iter, sync_mode, batch_size);

    if (sync_mode) {
      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
    }

    if (rank == 0 and (i + 1) % test_interval == 0) {
      std::string filename = root + "/test/part-001";
      distlr::DataIter test_iter(filename, num_feature_dim);
      lr.Test(test_iter, i + 1);
    }
  }

  if (rank == 0) {
    // duration
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    u_int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start).count();
    std::cout << "The training time is "<< elapsed_ms << std::endl;
  }

  std::string modelfile = root + "/models/part-00" + std::to_string(rank + 1);
  lr.SaveModel(modelfile);
}

int main() {
  StartServer();

  ps::Start();
  RunWorker();

  ps::Finalize();
  return 0;
}
