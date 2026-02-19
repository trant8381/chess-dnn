#include "constants.h"
#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <sys/types.h>
#include <vector>
#include <sstream>

struct GameStats {
  std::string fen;
  std::vector<int32_t> distribution;
  float result;

  GameStats() {};
  GameStats(std::string _fen, std::vector<int32_t> _distribution,
            float _result)
      : fen(_fen), distribution(_distribution), result(_result) {}
};

inline std::ostream& operator<<(std::ostream& stream, GameStats stats) {
    std::stringstream ss;
    for (auto it = stats.distribution.begin(); it != stats.distribution.end(); it++)
        ss << *it << " ";
    stream << std::format("{}\n{}\n{}", stats.fen, ss.str(), stats.result);
    return stream;
  }

class PlayBuffer {
private:
  std::array<GameStats, BUFFER_SIZE> arr;
  bool complete = false;
  int index = 0;
  uint64_t _totalSize;

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<> dist;

public:
  std::mutex lock;
  PlayBuffer() {
    gen = std::mt19937(rd());
    dist = std::uniform_int_distribution<>(1, BUFFER_SIZE / SAMPLE_SIZE);
  }

  void insert(GameStats gameStats) {
    _totalSize += 1;
    if (index == BUFFER_SIZE) {
      index = 0;
      arr[0] = gameStats;
      complete = true;
    } else {
      arr[index] = gameStats;
      index += 1;
    }
  }

  size_t size() {
    if (complete) {
      return BUFFER_SIZE;
    } else {
      return index;
    }
  }

  uint64_t totalSize() {
    return _totalSize;
  }

  std::vector<GameStats> sample() {
    std::vector<GameStats> samp;

    for (size_t i = 0; i < this->size(); i++) {
      int num = dist(gen);
      if (num == 1) {
        samp.push_back(arr[i]);
      }
    }

    return samp;
  }
};