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
#include <dirent.h>
#include <functional>
#include <algorithm>

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

void SplitFilename (const string& str, string& folder, string& prefix)
{
  size_t found;
  found = str.find_last_of("/\\");
  folder = str.substr(0, found);
  prefix = str.substr(found + 1);
}

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
        int msec = util::ToInt(ps::Environment::Get()->find("TAU"));
        tau_ = msec / 1000;
        ntau_ = ((__syscall_slong_t)(msec - tau_*1000)) * 1000000;
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
    // initialize update
    update_ = VectorXd::Zero(ndims_);

    accumulated_grad_.initialized = false;

    // save model
    if (!eval_) {
      time_t rawtime;
      struct tm *timeinfo;
      char buffer[80];
      time(&rawtime);
      timeinfo = localtime(&rawtime);
      strftime (buffer, 80, "%Y%m%d%H%M%S", timeinfo);
      save_filename_ = ps::Environment::Get()->find("SAVE_PREFIX") + string("sync") + to_string(sync_mode_) + string("_") + string(buffer);
      cout << save_filename_ << endl;
    }

    start_time_ = std::chrono::system_clock::now();

    eval_ = (util::ToInt(ps::Environment::Get()->find("EVAL")) == 1);
    if (eval_) {
      cout << "eval mode!" << endl;
      eval_file_.open(ps::Environment::Get()->find("EVAL_FILE"));
      eval_file_ >> eval_usec_;
      for (int i = 0; i < ndims_; i++) {
        eval_file_ >> weight_(i);
      }
      eval_file_output_.open(ps::Environment::Get()->find("EVAL_FILE") + string("_eval"), std::ofstream::out);
      cout << ps::Environment::Get()->find("EVAL_FILE") << endl;
    }
    accumulated_eval_.eval = 0;
    accumulated_eval_.naggregates = 0;

  }

  ~KVStoreDistServer() {
    if (ps_server_) {
      delete ps_server_;
    }
  }

private:

  // timer
  static void *SyncTimer(void *ptr) {
    // only for DGD-NOVR

    while (true) {
      pthread_mutex_lock(&timer_mutex_);
      if (global_ts_ == 0) {
        pthread_cond_wait(&timer_cond_, &timer_mutex_);
      }
      else {
        // absolute time!
        struct timespec time_to_wait;
        timespec_get(&time_to_wait, TIME_UTC);
        time_to_wait.tv_sec += tau_;
        time_to_wait.tv_nsec += ntau_;
        pthread_cond_timedwait(&timer_cond_, &timer_mutex_, &time_to_wait);
      }
      pthread_mutex_unlock(&timer_mutex_);

      pthread_mutex_lock(&weight_mutex_);

      // timeout, then apply gradient and synchronize
      // update the weight
      // gradient descent
      auto &merged = accumulated_grad_;
//      cout << " Iteration "<< global_ts_ << ", received: " << merged.naggregates << endl;
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
          response.ts1 = global_ts_ - 1;
        }
        response.ts2 = global_ts_;

        ps_server_->Response(pull_req.req_meta, response);
      }
      // erase
      pull_buf.clear();

      // read testing data
      if (ps::Environment::Get()->find("TEST_FILE") != nullptr) {
        std::string test_filename = ps::Environment::Get()->find("TEST_FILE");
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
      }

      // save model
      std::ofstream weight_file;
      Eigen::IOFormat CleanFmt(Eigen::FullPrecision, 0, ", ", "\t");
      std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
      u_int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time_).count();
      weight_file.open(save_filename_, std::ofstream::out | std::ofstream::app);
      weight_file << elapsed_ms << "\t" << weight_.format(CleanFmt) << endl;
      weight_file.close();

      std::cout << " Iteration " << global_ts_ << std::endl;

      pthread_mutex_unlock(&weight_mutex_);

      if (global_ts_ == num_iteration_) {
        // termination signal
        break;
      }
    }

    return NULL;
  }

  static void *SyncTimerUpdate(void *ptr) {
    // only for DGD-VR

    while (true) {
      pthread_mutex_lock(&timer_mutex_);
      if (global_ts_ == 0) {
        pthread_cond_wait(&timer_cond_, &timer_mutex_);
      }
      else {
        // absolute time!
        struct timespec time_to_wait;
        timespec_get(&time_to_wait, TIME_UTC);
        time_to_wait.tv_sec += tau_;
        time_to_wait.tv_nsec += ntau_;
        pthread_cond_timedwait(&timer_cond_, &timer_mutex_, &time_to_wait);
      }
      pthread_mutex_unlock(&timer_mutex_);

      pthread_mutex_lock(&weight_mutex_);

      // timeout, then apply gradient and synchronize
      // update the weight
      // gradient descent
      auto &merged = accumulated_grad_;
//      cout << " Iteration "<< global_ts_ << ", received: " << merged.naggregates << endl;
      weight_ -= learning_rate_ * (merged.vals / merged.naggregates + update_ / nsamples_);
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
      // renew update
      update_ = update_.eval() + merged.vals;
      // timestamp
      global_ts_++;
      // clear
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
          response.ts1 = update_tracker[pull_req.req_meta.sender];
        }
        response.ts2 = global_ts_;

        ps_server_->Response(pull_req.req_meta, response);
      }
      // erase
      pull_buf.clear();

      // read testing data
      if (ps::Environment::Get()->find("TEST_FILE") != nullptr) {
        std::string test_filename = ps::Environment::Get()->find("TEST_FILE");
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
      }

      // save model
      std::ofstream weight_file;
      Eigen::IOFormat CleanFmt(Eigen::FullPrecision, 0, ", ", "\t");
      std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
      u_int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time_).count();
      weight_file.open (save_filename_, std::ofstream::out | std::ofstream::app);
      weight_file << elapsed_ms << "\t" << weight_.format(CleanFmt) << endl;
      weight_file.close();

      std::cout << " Iteration " << global_ts_ << std::endl;

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

    if (eval_) {
      //// eval mode
      if (req_meta.push) {

        auto &merged = accumulated_eval_;

        merged.eval += req_data.vals[0];
        // the last element is the number of gradients
        merged.naggregates += req_data.vals[1];
//            if (req_data.vals[n]!=100) {
//              std::cout << "batchsize: " << req_data.vals[n] << std::endl;
//              std::cout << "total size: " << merged.naggregates  << std::endl;
//            }

        // synchronization
        // TODO: use the number of workers
        if (merged.naggregates == nsamples_) {
          // timestamp
          global_ts_++;
          merged.eval = merged.eval / merged.naggregates;
          // proximal
          if (use_proximal_) {
            if (proximal_op_ == 1) {
              // l1 proximal
              merged.eval += prox_opl1.cost(weight_);
            }
            else if (proximal_op_ == 2) {
              // l2 proximal
              merged.eval += prox_opl2.cost(weight_);
            }
          }
          std::cout << " Iteration "<< global_ts_ << ", time: " << std::setw(8) << eval_usec_ << ", cost: " << std::setw(8) << merged.eval << std::endl;
          eval_file_output_ << " Iteration "<< global_ts_ << ", time: " << std::setw(8) << eval_usec_ << ", cost: " << std::setw(8) << merged.eval << std::endl;
          merged.eval = 0;
          merged.naggregates = 0;
          if (global_ts_ == num_iteration_) {
            eval_file_output_.close();
            eval_file_.close();
          }
          else {
            // update weight
            eval_file_ >> eval_usec_;
            for (int i = 0; i < ndims_; i++) {
              eval_file_ >> weight_(i);
            }
          }
        }
        server->Response(req_meta);
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
        response.ts2 = global_ts_;
        server->Response(req_meta, response);
        // TODO: semi-synchronous
      }
    }
    else if (sync_mode_ == 1) {
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

        // synchronization
        // TODO: use the number of workers
//        cout << merged.naggregates << endl;
        if (merged.naggregates == nsamples_) {
//              cout << "apply gradients" << endl;
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

          // save model
          std::ofstream weight_file;
          Eigen::IOFormat CleanFmt(Eigen::FullPrecision, 0, ", ", "\t");
          std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
          u_int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time_).count();
          weight_file.open (save_filename_, std::ofstream::out | std::ofstream::app);
          weight_file << elapsed_ms << "\t" << weight_.format(CleanFmt) << endl;
          weight_file.close();

          std::cout << " Iteration " << global_ts_ << std::endl;
        }

        server->Response(req_meta);

        // read testing data
        if (show_test && ps::Environment::Get()->find("TEST_FILE") != nullptr) {
          std::string test_filename = ps::Environment::Get()->find("TEST_FILE");
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
        else {
//          std::cout << " Iteration " << global_ts_ << std::endl;
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
        response.ts2 = global_ts_;
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
        if (ts2 == global_ts_) {
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
          if (merged.naggregates == nsamples_) {
            // trigger
            pthread_cond_broadcast(&timer_cond_);
          }
        }

        pthread_mutex_unlock(&weight_mutex_);

      } else { // pull
        CHECK(weight_initialized_);

        // TODO: special case: first pull

        pthread_mutex_lock(&weight_mutex_);
        if (global_ts_ == num_iteration_) {
          // terminate signal
          ps::KVPairs<Val> response;
          response.keys = req_data.keys;
          response.vals.resize(n);
          for (size_t i = 0; i < n; ++i) {
            response.vals[i] = weight_(i);
          }
          // timestamp
          // should be 0
          response.ts1 = -1;
          response.ts2 = global_ts_;
          server->Response(req_meta, response);
        }
        else if (pull_tracker.count(req_meta.sender) == 0) {
          // first pull
          pull_tracker[req_meta.sender] = 1;
          ps::KVPairs<Val> response;
          response.keys = req_data.keys;
          response.vals.resize(n);
          for (size_t i = 0; i < n; ++i) {
            response.vals[i] = weight_(i);
          }
          // timestamp
          // should be 0
          response.ts1 = global_ts_;
          response.ts2 = global_ts_;
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
        if (ts2 == global_ts_ && ts1 == update_tracker[req_meta.sender]) {
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

          // update the tracker
          update_tracker[req_meta.sender] = global_ts_;

          // synchronization
          if (merged.naggregates == nsamples_) {
            // trigger
            pthread_cond_broadcast(&timer_cond_);
          }
        }

        pthread_mutex_unlock(&weight_mutex_);

      } else { // pull
        CHECK(weight_initialized_);

        pthread_mutex_lock(&weight_mutex_);
        if (global_ts_ == num_iteration_) {
          // terminate signal
          ps::KVPairs<Val> response;
          response.keys = req_data.keys;
          response.vals.resize(n);
          for (size_t i = 0; i < n; ++i) {
            response.vals[i] = weight_(i);
          }
          // timestamp
          // should be 0
          response.ts1 = -1;
          response.ts2 = global_ts_;
          server->Response(req_meta, response);
        }
        else if (pull_tracker.count(req_meta.sender) == 0) {
          // first pull
          pull_tracker[req_meta.sender] = 1;
          ps::KVPairs<Val> response;
          response.keys = req_data.keys;
          response.vals.resize(n);
          for (size_t i = 0; i < n; ++i) {
            response.vals[i] = weight_(i);
          }
          // timestamp
          // should be 0
          response.ts1 = global_ts_;
          response.ts2 = global_ts_;
          // initialize update_tracker
          update_tracker[req_meta.sender] = 0;
          server->Response(req_meta, response);

          if (pull_tracker.size() == ps::NumWorkers()) {
            // TODO: use another handler for DGD-VR
            pthread_create(&timer_thread_, NULL, this->SyncTimerUpdate, NULL);
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

  }

  int sync_mode_;
  static int num_iteration_;
  static double learning_rate_;
  // timestamp for iterations
  static int global_ts_;
  static int nsamples_;
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
  static __time_t tau_;
  static __syscall_slong_t ntau_;
  static std::map<int, int> pull_tracker;
  static VectorXd update_;
  static std::map<int, int> update_tracker;
  static pthread_mutex_t timer_mutex_;
  static pthread_mutex_t weight_mutex_;
  static pthread_cond_t timer_cond_;

  static string save_filename_;

  static std::chrono::time_point<std::chrono::system_clock> start_time_;

  // evaluation
  bool eval_;
  struct MergeEval {
    double eval;
    // number of aggregates
    int naggregates;
  };
  MergeEval accumulated_eval_;
  std::ifstream eval_file_;
  std::ofstream eval_file_output_;
  int eval_usec_;

};

template <typename Val>
int KVStoreDistServer<Val>::num_iteration_;
template <typename Val>
double KVStoreDistServer<Val>::learning_rate_;
// timestamp for iterations
template <typename Val>
int KVStoreDistServer<Val>::global_ts_;
template <typename Val>
int KVStoreDistServer<Val>::nsamples_;
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
__time_t KVStoreDistServer<Val>::tau_;
template <typename Val>
__syscall_slong_t KVStoreDistServer<Val>::ntau_;
template <typename Val>
std::map<int, int> KVStoreDistServer<Val>::pull_tracker;
template <typename Val>
VectorXd KVStoreDistServer<Val>::update_;
template <typename Val>
std::map<int, int> KVStoreDistServer<Val>::update_tracker;
template <typename Val>
pthread_mutex_t KVStoreDistServer<Val>::timer_mutex_;
template <typename Val>
pthread_mutex_t KVStoreDistServer<Val>::weight_mutex_;
template <typename Val>
pthread_cond_t KVStoreDistServer<Val>::timer_cond_;

template <typename Val>
string KVStoreDistServer<Val>::save_filename_;
template <typename Val>
std::chrono::time_point<std::chrono::system_clock> KVStoreDistServer<Val>::start_time_;

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
  // only for DGD-NOVR
  PushPackage *push_package = (PushPackage*)ptr;
  // gradient
  VectorXd::Map(&(push_package->vec_weight_push->at(0)), push_package->ndims) = push_package->lr->grad(push_package->dr->getX(), push_package->dr->gety());
  // timestamps
  push_package->vec_weight_push->at(push_package->ndims+1) = push_package->ts1;
  push_package->vec_weight_push->at(push_package->ndims+2) = push_package->ts2;
  // push, no wait
  // TODO: simulate the delay
  if (((double) rand() / (RAND_MAX)) < push_package->delay_prob) {
    usleep(push_package->delay_usec);
  }
  push_package->kv->Push(*(push_package->keys_push), *(push_package->vec_weight_push));
  return NULL;
}

struct UpdatePackage {
  lrprox::LR *lr;
  vector<double> *vec_weight_push;
  ps::KVWorker<double>* kv;
  std::vector<ps::Key> *keys_push;
  lrprox::data_reader *dr;
  int ndims;
  int ts1;
  int ts2;
  std::map<int, VectorXd> *grad_tracker;
  int *localts;
  double delay_prob;
  int delay_usec;
};

void *ComputePushUpdate(void *ptr) {
  // only for DGD-VR
  UpdatePackage *update_package = (UpdatePackage*)ptr;
  // compute gradient and storage
//  (*(update_package->grad_tracker))[update_package->ts2] = update_package->lr->grad(update_package->dr->getX(), update_package->dr->gety());
  update_package->grad_tracker->insert(std::pair<int, VectorXd>(update_package->ts2, update_package->lr->grad(update_package->dr->getX(), update_package->dr->gety())));
  *(update_package->localts) = update_package->ts2 + 1;
  // delete cache
  std::vector<int> ts_to_delete;
  for (auto const &cache : *(update_package->grad_tracker)) {
    if (cache.first < update_package->ts1) {
      ts_to_delete.push_back(cache.first);
    }
  }
  for (auto const &ts : ts_to_delete) {
    update_package->grad_tracker->erase(ts);
  }
  // compute the update
  if (update_package->ts2 == 0) {
    VectorXd::Map(&(update_package->vec_weight_push->at(0)), update_package->ndims) = update_package->grad_tracker->at(update_package->ts2);
  }
  else {
//    cerr << update_package->ts1 << ", " << update_package->ts2 << endl;
    VectorXd::Map(&(update_package->vec_weight_push->at(0)), update_package->ndims) = update_package->grad_tracker->at(update_package->ts2) - update_package->grad_tracker->at(update_package->ts1);
  }
  // send the update
  // timestamps
  update_package->vec_weight_push->at(update_package->ndims+1) = update_package->ts1;
  update_package->vec_weight_push->at(update_package->ndims+2) = update_package->ts2;
  // push, no wait
  // TODO: simulate the delay
  if (((double) rand() / (RAND_MAX)) < update_package->delay_prob) {
    usleep(update_package->delay_usec);
  }
  update_package->kv->Push(*(update_package->keys_push), *(update_package->vec_weight_push));
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

  // train data prefix
  std::string root = ps::Environment::Get()->find("TRAIN_DIR");
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
  string folder, prefix;
  SplitFilename(root, folder, prefix);
  vector<string> filelist;
  vector<string> filelist_local;
  DIR *dpdf;
  struct dirent *epdf;
  dpdf = opendir(folder.c_str());
  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
      if (strncmp(epdf->d_name, prefix.c_str(), prefix.length()) == 0) {
        filelist.push_back(folder + "/" + string(epdf->d_name));
      }
    }
  }
  std::sort(filelist.begin(), filelist.end(), [](const string& a, const string& b) {
    hash<string> hasher;
    return hasher(b) < hasher(a);
  });
  for (int i = 0; i < filelist.size(); i++) {
    if (i % ps::NumWorkers() == ps::MyRank()) {
      filelist_local.push_back(filelist[i]);
//      cout << "Worker[" << rank << "]:" << filelist[i] << endl;
    }
  }
  lrprox::data_reader dr = lrprox::data_reader(filelist_local, nfeatures);
//  cout << "Worker[" << rank << "]:" << "Local data size:" << dr.getX().rows() << endl;

  int ts1 = 0, ts2 = 0;

  bool eval = (util::ToInt(ps::Environment::Get()->find("EVAL")) == 1);

  if (eval) {
    //// eval mode

    std::vector<ps::Key> keys_eval(2);
    for (size_t i = 0; i < keys_eval.size(); ++i) {
      keys_eval[i] = i;
      keys_eval[i] = i;
    }
    vector<double> vec_eval_push(keys_eval.size());

    while(true) {

      // pull
      kv->Wait(kv->Pull(keys_pull, &vec_weight_pull, nullptr, 0, nullptr, &ts1, &ts2));
      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
      // termination
      if (ts1 == -1) {
        break;
      }
      // copy to eigen
      // TODO: improvement?
      lr.updateWeight(vec_weight_pull);
      // eval
      vec_eval_push[0] = lr.cost(dr.getX(), dr.gety());
      // naggregates
      vec_eval_push[1] = dr.getX().rows();
      // push
      kv->Wait(kv->Push(keys_eval, vec_eval_push));

      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
    }
  }
  else if (sync_mode == 1) {
    //// sync mode
    while(true) {

      // pull
      kv->Wait(kv->Pull(keys_pull, &vec_weight_pull, nullptr, 0, nullptr, &ts1, &ts2));
      if (((double) rand() / (RAND_MAX)) < delay_prob) {
        usleep(delay_usec);
      }
      ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
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
      if (((double) rand() / (RAND_MAX)) < delay_prob) {
        usleep(delay_usec);
      }
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
    keys_push.push_back(ndims+1);
    keys_push.push_back(ndims+2);
    vec_weight_push.push_back(0);
    vec_weight_push.push_back(0);

    UpdatePackage update_package;
    update_package.keys_push = &keys_push;
    update_package.kv = kv;
    update_package.lr = &lr;
    update_package.vec_weight_push = &vec_weight_push;
    update_package.dr = &dr;
    update_package.ndims = ndims;
    update_package.delay_prob = delay_prob;
    update_package.delay_usec = delay_usec;
    // naggregates
    vec_weight_push[ndims] = dr.getX().rows();
    pthread_t update_thread;
    bool grad_thread_initialized = false;
    std::map<int, VectorXd> grad_tracker;
    int localts = 0;
    update_package.localts = &localts;
    update_package.grad_tracker = &grad_tracker;
    while(true) {

      // pull
      kv->Wait(kv->Pull(keys_pull, &vec_weight_pull, nullptr, 0, nullptr, &ts1, &ts2));
      if (((double) rand() / (RAND_MAX)) < delay_prob) {
        usleep(delay_usec);
      }
      // termination
      if (ts1 == -1) {
        pthread_cancel(update_thread);
        break;
      }
      // drop old message, not sure if really necessary
      if (ts2 < localts) {
        continue;
      }
      // copy to eigen
      lr.updateWeight(vec_weight_pull);

      if (grad_thread_initialized) {
        pthread_cancel(update_thread);
      }

      update_package.ts1 = ts1;
      update_package.ts2 = ts2;
      pthread_create(&update_thread, NULL, ComputePushUpdate, &update_package);
      pthread_detach(update_thread);
      grad_thread_initialized = true;

    }
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