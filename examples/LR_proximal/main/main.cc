#include "ps/ps.h"
#include <iostream>
#include <vector>
#include <map>
#include "util.h"
#include "data_reader.h"
#include "lr.h"
#include "prox_l1.h"
#include "prox_l2.h"

#include <Eigen/Dense>
#include <iomanip>
#include <unistd.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;

using namespace std;
//using namespace ps;

struct MergeBuf {
  VectorXd vals;
  // number of aggregates
  int naggregates;
  bool initialized;
};
template <typename Val>
struct PullBuf {
  ps::KVMeta req_meta;
  ps::KVPairs<Val> response;
};

template <typename Val>
class KVStoreDistServer {
public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<double>(0);
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));

    // read environment variables
    if (ps::Environment::Get()->find("SYNC_MODE") != nullptr) {
      sync_mode_ = util::ToInt(ps::Environment::Get()->find("SYNC_MODE"));
      if (sync_mode_ == 2 || sync_mode_ == 3) {
        tau_ = util::ToInt(ps::Environment::Get()->find("TAU"));
      }
    }
    else {
      // default: sync mode
      sync_mode_ = 1;
    }
    num_iteration_ = util::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
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

    std::vector<std::string> mode = {"sync", "semi-sync without VR", "semi-sync with VR"};
    std::cout << "Server mode: " << mode[sync_mode_-1] << std::endl;

    // initialize timestamp
    global_ts_ = 0;

    // initialize the timer
    pthread_mutex_init(&timer_mutex_, NULL);
    pthread_mutex_init(&weight_mutex_, NULL);
    pthread_cond_init(&timer_cond_, NULL);


    // TODO: initialize weights
    weight_ = VectorXd::Ones(ndims_);
    weight_initialized_ = true;
  }

  ~KVStoreDistServer() {
    if (ps_server_) {
      delete ps_server_;
    }
  }

private:

  // timer
  static void *SyncTimer(void *ptr) {

    struct timespec time_to_wait = {0, tau_};

    while (true) {
      pthread_mutex_lock(&timer_mutex_);
      pthread_cond_timedwait(&timer_cond_, &timer_mutex_, &time_to_wait);
      pthread_mutex_unlock(&timer_mutex_);

      pthread_mutex_lock(&weight_mutex_);

      // timeout, then apply gradient and synchronize
      // update the weight
      // gradient descent
      auto &merged = accumulated_grad_;
      cout << " Iteration "<< global_ts_ << ", received: " << merged.naggregates << endl;
      weight_ -= learning_rate_ * merged.vals / merged.naggregates;
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
      merged.vals.setZero(ndims_);
      merged.naggregates = 0;
      // TODO: pull buffer
      ps::KVPairs<Val> response;
      response.vals.resize(ndims_);
      for (size_t i = 0; i < ndims_; ++i) {
        response.vals[i] = weight_(i);
      }
      for (auto const &pull_req : pull_buf) {
        // TODO: for one single server, the keys are not necessary for pull
        response.keys = pull_req.response.keys;
        // timestamp
        // note: update timestamp only for synchronous mode

        if (global_ts_ == num_iteration_) {
          // termination signal
          response.ts1 = -1;
        }
        else {
          response.ts1 = global_ts_;
        }
        response.ts2 = global_ts_ + 1;

        ps_server_->Response(pull_req.req_meta, response);
      }
      // erase
      pull_buf.clear();

      // read testing data
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

      pthread_mutex_unlock(&weight_mutex_);

      if (global_ts_ == num_iteration_) {
        // termination signal
        break;
      }
    }

    return NULL;
  }

  // threadsafe
  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<Val>& req_data,
                  ps::KVServer<Val>* server) {
    // currently, only support one server
    
    size_t n = ndims_;
    bool show_test = false;

    if (sync_mode_ == 1) {
      //// sync mode
      if (req_meta.push) {

        CHECK_EQ(n+1, req_data.vals.size());
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

        if (global_ts_ == num_iteration_) {
          // termination signal
          response.ts1 = -1;
        }
        else {
          response.ts1 = global_ts_;
        }
        response.ts2 = global_ts_ + 1;
        server->Response(req_meta, response);
        // TODO: semi-synchronous
      }
    }
    else if(sync_mode_ == 2) {
      //// semi-sync without vr
      if (req_meta.push) {

        // 3 additional value: naggregates, ts1, ts2
        CHECK_EQ(n+3, req_data.vals.size());
        // response
        server->Response(req_meta);

        pthread_mutex_lock(&weight_mutex_);
        // drop old message
        int ts1, ts2;
        ts1 = req_data.vals[n+1];
        ts2 = req_data.vals[n+2];
        if (ts2 == global_ts_ + 1) {
          // valid push
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

          // synchronization
//          if (merged.naggregates == nsamples_) {
//            // trigger
//            pthread_cond_broadcast(&timer_cond_);
//          }
        }

        pthread_mutex_unlock(&weight_mutex_);

      } else { // pull
        CHECK(weight_initialized_);

        // TODO: special case: first pull

        pthread_mutex_lock(&weight_mutex_);
        if (pull_tracker.count(req_meta.sender) == 0) {
          // first pull
          pull_tracker[req_meta.sender] = 1;
          ps::KVPairs<Val> response;
          response.keys = req_data.keys;
          response.vals.resize(n);
          for (size_t i = 0; i < n; ++i) {
            response.vals[i] = weight_(i);
          }
          // timestamp
          response.ts1 = global_ts_;
          response.ts2 = global_ts_ + 1;
          server->Response(req_meta, response);

          if (pull_tracker.size() == ps::NumWorkers()) {
            pthread_create(&timer_thread_, NULL, this->SyncTimer, NULL);
            pthread_detach(timer_thread_);
          }
        }
        else {
          PullBuf<Val> pull_req;
          pull_req.response.keys = req_data.keys;
          pull_req.req_meta = req_meta;

          // buffer
          pull_buf.push_back(pull_req);

          pull_tracker[req_meta.sender] = pull_tracker[req_meta.sender] + 1;
        }
        pthread_mutex_unlock(&weight_mutex_);

      }
    }
    else if(sync_mode_ == 3) {
      //// semi-sync with vr
    }

  }

  int sync_mode_;
  static int num_iteration_;
  static double learning_rate_;
  // timestamp for iterations
  static int global_ts_;
  int nsamples_;
  static int ndims_;
  bool weight_initialized_;
  static bool use_proximal_;
  static int proximal_op_;
  static lrprox::prox_l1 prox_opl1;
  static lrprox::prox_l2 prox_opl2;

  static VectorXd weight_;
  std::unordered_map<int, VectorXd> weights_;
  static MergeBuf accumulated_grad_;
  static ps::KVServer<double>* ps_server_;
  pthread_t timer_thread_;

  // for semi-sync mode
  static std::vector<PullBuf<Val>> pull_buf;
  static int tau_;
  static std::map<int, int> pull_tracker;
  static pthread_mutex_t timer_mutex_;
  static pthread_mutex_t weight_mutex_;
  static pthread_cond_t timer_cond_;

};

template <typename Val>
int KVStoreDistServer<Val>::num_iteration_;
template <typename Val>
double KVStoreDistServer<Val>::learning_rate_;
// timestamp for iterations
template <typename Val>
int KVStoreDistServer<Val>::global_ts_;
template <typename Val>
int KVStoreDistServer<Val>::ndims_;
template <typename Val>
bool KVStoreDistServer<Val>::use_proximal_;
template <typename Val>
int KVStoreDistServer<Val>::proximal_op_;
template <typename Val>
lrprox::prox_l1 KVStoreDistServer<Val>::prox_opl1;
template <typename Val>
lrprox::prox_l2 KVStoreDistServer<Val>::prox_opl2;

template <typename Val>
VectorXd KVStoreDistServer<Val>::weight_;
template <typename Val>
MergeBuf KVStoreDistServer<Val>::accumulated_grad_;
template <typename Val>
ps::KVServer<double>* KVStoreDistServer<Val>::ps_server_;

template <typename Val>
std::vector<PullBuf<Val>> KVStoreDistServer<Val>::pull_buf;
template <typename Val>
int KVStoreDistServer<Val>::tau_;
template <typename Val>
std::map<int, int> KVStoreDistServer<Val>::pull_tracker;
template <typename Val>
pthread_mutex_t KVStoreDistServer<Val>::timer_mutex_;
template <typename Val>
pthread_mutex_t KVStoreDistServer<Val>::weight_mutex_;
template <typename Val>
pthread_cond_t KVStoreDistServer<Val>::timer_cond_;

void StartServer() {
  if (!ps::IsServer()) {
    return;
  }
  auto server = new KVStoreDistServer<double>();
  ps::RegisterExitCallback([server](){ delete server; });
}

struct PushPackage {
  lrprox::LR *lr;
  vector<double> *vec_weight_push;
  ps::KVWorker<double>* kv;
  std::vector<ps::Key> *keys_push;
  lrprox::data_reader *dr;
  int ndims;
  int ts1;
  int ts2;
  double delay_prob;
  int delay_usec;
};

void *ComputePushGrad(void *ptr) {
  PushPackage *push_package = (PushPackage*)ptr;
  // gradient
  VectorXd::Map(&(push_package->vec_weight_push->at(0)), push_package->ndims) = push_package->lr->grad(push_package->dr->getX(), push_package->dr->gety());
  // timestamps
  push_package->vec_weight_push->at(push_package->ndims+1) = push_package->ts1;
  push_package->vec_weight_push->at(push_package->ndims+2) = push_package->ts2;
  // push, no wait
  // TODO: simulate the delay
  if (((double) rand() / (RAND_MAX)) < push_package->delay_prob) {
    cout << "delayed!" << endl;
    usleep(push_package->delay_usec);
  }
  push_package->kv->Push(*(push_package->keys_push), *(push_package->vec_weight_push));
  return NULL;
}

void RunWorker() {
  if (!ps::IsWorker()) {
    return;
  }

  int sync_mode;

  if (ps::Environment::Get()->find("SYNC_MODE") != nullptr) {
    sync_mode = util::ToInt(ps::Environment::Get()->find("SYNC_MODE"));
  }
  else {
    // default: sync mode
    sync_mode = 1;
  }

  // data folder
  std::string root = ps::Environment::Get()->find("DATA_DIR");
  int nfeatures = util::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));
  int ndims = nfeatures + 1;

  // simulate message delay
  double delay_prob = util::ToDouble(ps::Environment::Get()->find("GD_DELAY_MSG")) / 100;
  // millisecond to microsecond
  int delay_usec = util::ToInt(ps::Environment::Get()->find("GD_RESEND_DELAY")) * 1000;
  srand (time(NULL) + ps::MyRank());

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

//  if (rank == 0) {
//    auto vals = lr.getWeight();
//    VectorXd::Map(&vec_weight_pull[0], ndims) = vals;
//    // the naggregates is not necessary for initialization push, use keys_pull
//    kv->Wait(kv->Push(keys_pull, vec_weight_pull));
//  }
//  ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);

  std::cout << "Worker[" << rank << "]: start working..." << std::endl;
//  int num_iteration = util::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));

  // add timer
  std::chrono::time_point<std::chrono::system_clock> start;
  if (rank == 0) {
    start = std::chrono::system_clock::now();
  }

  // read training data
  // TODO: psuedo-distirbuted environment
  std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
  lrprox::data_reader dr = lrprox::data_reader(filename, nfeatures);

  int ts1 = 0, ts2 = 0;

  if (sync_mode == 1) {
    //// sync mode
    while(true) {

      // pull
      kv->Wait(kv->Pull(keys_pull, &vec_weight_pull, nullptr, 0, nullptr, &ts1, &ts2));
      // termination
      if (ts1 == -1) {
        break;
      }
      // copy to eigen
      // TODO: improvement?
      lr.updateWeight(vec_weight_pull);
      // gradient
      VectorXd::Map(&vec_weight_push[0], ndims) = lr.grad(dr.getX(), dr.gety());
      // naggregates
      vec_weight_push[ndims] = dr.getX().rows();
      // push
      if (((double) rand() / (RAND_MAX)) < delay_prob) {
        cout << "delayed!" << endl;
        usleep(delay_usec);
      }
      kv->Wait(kv->Push(keys_push, vec_weight_push));

      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
    }
  }
  else if (sync_mode == 2) {
    //// semi-sync without vr
    keys_push.push_back(ndims+1);
    keys_push.push_back(ndims+2);
    vec_weight_push.push_back(0);
    vec_weight_push.push_back(0);

    PushPackage push_package;
    push_package.keys_push = &keys_push;
    push_package.kv = kv;
    push_package.lr = &lr;
    push_package.vec_weight_push = &vec_weight_push;
    push_package.dr = &dr;
    push_package.ndims = ndims;
    push_package.delay_prob = delay_prob;
    push_package.delay_usec = delay_usec;
    // naggregates
    vec_weight_push[ndims] = dr.getX().rows();
    pthread_t grad_thread;
    bool grad_thread_initialized = false;
    while(true) {

      // pull
      kv->Wait(kv->Pull(keys_pull, &vec_weight_pull, nullptr, 0, nullptr, &ts1, &ts2));
      // termination
      if (ts1 == -1) {
        pthread_cancel(grad_thread);
        break;
      }
      // copy to eigen
      lr.updateWeight(vec_weight_pull);

      if (grad_thread_initialized) {
        pthread_cancel(grad_thread);
      }

      push_package.ts1 = ts1;
      push_package.ts2 = ts2;
      pthread_create(&grad_thread, NULL, ComputePushGrad, &push_package);
      pthread_detach(grad_thread);
      grad_thread_initialized = true;

    }
  }
  else if (sync_mode == 3) {
    //// semi-sync with vr
  }

  ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
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

#pragma clang diagnostic pop