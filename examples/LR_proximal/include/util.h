//
// Created by cx2 on 3/20/17.
//

#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <vector>

namespace util {

  std::vector<std::string> Split(std::string line, char sparator) {
    std::vector<std::string> ret;

    int start = 0;
    std::size_t pos = line.find(sparator, start);
    while (pos != std::string::npos) {
      ret.push_back(line.substr(start, pos));
      start = pos + 1;
      pos = line.find(sparator, start);
    }
    ret.push_back(line.substr(start));
    return ret;
  }

  int ToInt(const char* str) {
    return std::stoi(std::string(str));
  }

  int ToInt(const std::string& str) {
    return std::stoi(str);
  }

  float ToDouble(const char* str){
    return std::stod(std::string(str));
  }

  float ToDouble(const std::string& str) {
    return std::stod(str);
  }

} // namespace util

#endif  // UTIL_H_