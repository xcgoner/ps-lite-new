#include "ps/ps.h"
#include <iostream>
#include <vector>
#include "util.h"
#include "data_reader.h"
#include "lr.h"
#include "prox_l1.h"
#include "prox_l2.h"

#include <Eigen/Dense>
#include <iomanip>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;

using namespace std;
//using namespace ps;

template <typename Val>
class KVStoreDistServer {
public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<double>(0);
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));

    // read environment variables
    sync_mode_ = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");
    learning_rate_ = util::ToDouble(ps::Environment::Get()->find("LEARNING_RATE"));
    nsamples_ = util::ToInt(ps::Environment::Get()->find("NSAMPLES"));
    // an addition column for all 1
    ndims_ = util::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM")) + 1;
    weight_initialized_ = false;
    // TODO: regularization
    auto proximal_op = ps::Environment::Get()->find("PROXIMAL");
    if (proximal_op != nullptr) {
      use_proximal_ = true;
      double lambda = util::ToDouble(ps::Environment::Get()->find("LAMBDA"));
      if (strcmp(proximal_op, "l1") == 0) {
        proximal_op_ = 1;
        prox_opl1 = lrprox::prox_l1(lambda);
      }
      else if (strcmp(proximal_op, "l2") == 0) {
        proximal_op_ = 2;
        prox_opl2 = lrprox::prox_l2(lambda);
      }
      cout << "Using proximal: " << proximal_op_ << endl;
    }
    else {
      use_proximal_ = false;
    }

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
    // currently, only support one server
    
    size_t n = ndims_;
    bool show_test = false;
    if (req_meta.push) {
      if (!weight_initialized_) {
        // initialization
        weight_initialized_ = true;
        CHECK_EQ(n, req_data.vals.size());
        std::cout << "Init weight" << std::endl;
        weight_ = VectorXd::Zero(n);
        for (int i = 0; i < n; ++i) {
          weight_(i) = req_data.vals[i];
        }
        server->Response(req_meta);
        if (sync_mode_) {
          accumulated_grad_.initialized = false;
        }
      } else {
        CHECK_EQ(n+1, req_data.vals.size());
        if (sync_mode_) {
          auto &merged = accumulated_grad_;
          // initialization
          if (!merged.initialized) {
            merged.initialized = true;
            merged.vals = VectorXd::Zero(n);
            merged.naggregates = 0;
          }

          for (int i = 0; i < n; ++i) {
            merged.vals(i) += req_data.vals[i];
          }
          // the last element is the number of gradients
          merged.naggregates += req_data.vals[n];
//            if (req_data.vals[n]!=100) {
//              std::cout << "batchsize: " << req_data.vals[n] << std::endl;
//              std::cout << "total size: " << merged.naggregates  << std::endl;
//            }

          server->Response(req_meta);
          // synchronization
          if (merged.naggregates == nsamples_) {
//              std::cout << "apply gradients" << std::endl;
            // update the weight
            // gradient descent
            weight_ -= learning_rate_ * merged.vals / merged.naggregates;
            // TODO: proximal step
            if (use_proximal_) {
              if (proximal_op_ == 1) {
                // l1 proximal
                weight_ = prox_opl1.proximal(weight_, learning_rate_);
              }
              else if (proximal_op_ == 2) {
                // l2 proximal
                weight_ = prox_opl2.proximal(weight_, learning_rate_);
              }
            }
            // timestamp
            global_ts_++;
            merged.vals.setZero(n);
            merged.naggregates = 0;
            show_test = true;
          }
        } else { // async push
          for (size_t i = 0; i < n; ++i) {
            // SGD
            weight_(i) -= learning_rate_ * req_data.vals[i] / req_data.vals[n];
            // TODO: proximal step
          }
          server->Response(req_meta);
        }
      }
      // read testing data
      if (show_test) {
        std::string root = ps::Environment::Get()->find("DATA_DIR");
        std::string test_filename = root + "/test/part-001";
        lrprox::data_reader test_dr = lrprox::data_reader(test_filename, ndims_-1);
        time_t rawtime;
        time(&rawtime);
        struct tm* curr_time = localtime(&rawtime);
        lrprox::LR lr = lrprox::LR(ndims_);
        lr.updateWeight(weight_);
        double cost = lr.cost(test_dr.getX(), test_dr.gety()) / test_dr.getX().rows();
        if (use_proximal_) {
          if (proximal_op_ == 1) {
            // l1 proximal
            cost = cost + prox_opl1.cost(weight_);
          }
          else if (proximal_op_ == 2) {
            // l2 proximal
            cost = cost + prox_opl2.cost(weight_);
          }
        }
        std::cout << std::setfill ('0') << std::setw(2) << curr_time->tm_hour << ':' << std::setfill ('0') << std::setw(2)
                  << curr_time->tm_min << ':' << std::setfill ('0') << std::setw(2) << curr_time->tm_sec
                  << " Iteration "<< global_ts_ << ", cost: " << cost
                  << std::endl;
        show_test = false;
      }
    } else { // pull
      CHECK(weight_initialized_);

      ps::KVPairs<Val> response;
      response.keys = req_data.keys;
      response.vals.resize(n);
      for (size_t i = 0; i < n; ++i) {
        response.vals[i] = weight_(i);
      }
      // timestamp
      // note: update timestamp only for synchronous mode
      response.ts1 = global_ts_;
      response.ts2 = global_ts_ + 1;
      server->Response(req_meta, response);
      // TODO: semi-synchronous
    }
  }

  bool sync_mode_;
  double learning_rate_;
  // timestamp for iterations
  int global_ts_;
  int nsamples_;
  int ndims_;
  bool weight_initialized_;
  bool use_proximal_;
  int proximal_op_;
  lrprox::prox_l1 prox_opl1;
  lrprox::prox_l2 prox_opl2;

  struct MergeBuf {
    VectorXd vals;
    // number of aggregates
    int naggregates;
    bool initialized;
  };

  VectorXd weight_;
  std::unordered_map<int, VectorXd> weights_;
  MergeBuf accumulated_grad_;
  ps::KVServer<double>* ps_server_;
};

void StartServer() {
  if (!ps::IsServer()) {
    return;
  }
  auto server = new KVStoreDistServer<double>();
  ps::RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!ps::IsWorker()) {
    return;
  }

  bool sync_mode = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");

  // data folder
  std::string root = ps::Environment::Get()->find("DATA_DIR");
  int nfeatures = util::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));
  int ndims = nfeatures + 1;

  int rank = ps::MyRank();
  // kv store
  ps::KVWorker<double>* kv = new ps::KVWorker<double>(0);
  lrprox::LR lr = lrprox::LR(ndims);

  // initialized the vector used to push
  vector<double> vec_weight_push(ndims+1);
  // initialized the vector used to pull
  vector<double> vec_weight_pull(ndims);

  // additional key for naggregates
  std::vector<ps::Key> keys_push(ndims+1);
  std::vector<ps::Key> keys_pull(ndims);
  for (size_t i = 0; i < keys_pull.size(); ++i) {
    keys_pull[i] = i;
    keys_push[i] = i;
  }
  // additional key for naggregates
  keys_push[ndims] = ndims;

  if (rank == 0) {
    auto vals = lr.getWeight();
    VectorXd::Map(&vec_weight_pull[0], ndims) = vals;
    // the naggregates is not necessary for initialization push, use keys_pull
    kv->Wait(kv->Push(keys_pull, vec_weight_pull));
  }
  ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);

  std::cout << "Worker[" << rank << "]: start working..." << std::endl;
  int num_iteration = util::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
  int batch_size = util::ToInt(ps::Environment::Get()->find("BATCH_SIZE"));
  int test_interval = util::ToInt(ps::Environment::Get()->find("TEST_INTERVAL"));

  // add timer
  std::chrono::time_point<std::chrono::system_clock> start;
  if (rank == 0) {
    start = std::chrono::system_clock::now();
  }

  // read training data
  // TODO: psuedo-distirbuted environment
  std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
  lrprox::data_reader dr = lrprox::data_reader(filename, nfeatures);

  int ts1, ts2;

  for (int i = 0; i < num_iteration; ++i) {

    // pull
    kv->Wait(kv->Pull(keys_pull, &vec_weight_pull, nullptr, 0, nullptr, &ts1, &ts2));
    // copy to eigen
    // TODO: improvement?
    lr.updateWeight(vec_weight_pull);
    // gradient
    VectorXd::Map(&vec_weight_push[0], ndims) = lr.grad(dr.getX(), dr.gety());
    // naggregates
    vec_weight_push[ndims] = dr.getX().rows();
    // push
    kv->Wait(kv->Push(keys_push, vec_weight_push));
    if (sync_mode) {
      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
    }

  }

  if (rank == 0) {
    // duration
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    u_int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start).count();
    std::cout << "The training time is "<< elapsed_ms << std::endl;
  }

//  std::string modelfile = root + "/models/part-00" + std::to_string(rank + 1);
//  lr.SaveModel(modelfile);
}

int main() {
  StartServer();

  ps::Start();
  RunWorker();

  ps::Finalize();
  return 0;
}
