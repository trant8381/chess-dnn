#include "dnn.h"
#include "mcts.h"
#include "move_gen.h"
#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <string>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <vector>

static std::vector<Midnight::Move> legalMoves(Midnight::Position board) {
  // Copy the moves while the MoveList is alive.
  if (board.turn() == Midnight::WHITE) {
    Midnight::MoveList<Midnight::WHITE> list(board);
    return std::vector<Midnight::Move>(list.begin(), list.end());
  }
  Midnight::MoveList<Midnight::BLACK> list(board);
  return std::vector<Midnight::Move>(list.begin(), list.end());
}

static void play(Midnight::Position &board, const Midnight::Move &move) {
  if (board.turn() == Midnight::WHITE) {
    board.play<Midnight::WHITE>(move);
  } else {
    board.play<Midnight::BLACK>(move);
  }
}

// UCI text for a move. Assumes a1 = 0 ... h8 = 63 and that the promotion piece
// is in the low two bits of type() (knight, bishop, rook, queen), which is
// what policyIndex in mcts.cpp assumes as well.
static std::string uci(const Midnight::Move &move) {
  const int from = static_cast<int>(move.from());
  const int to = static_cast<int>(move.to());

  std::string text;
  text += static_cast<char>('a' + from % 8);
  text += static_cast<char>('1' + from / 8);
  text += static_cast<char>('a' + to % 8);
  text += static_cast<char>('1' + to / 8);
  if (move.is_promotion()) {
    text += "nbrq"[move.type() & 0b0011];
  }
  return text;
}

// Index of the legal move written as `text`, or -1.
static int findMove(const std::vector<Midnight::Move> &moves,
                    std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  for (int pass = 0; pass < 2; pass++) {
    for (size_t i = 0; i < moves.size(); i++) {
      if (uci(moves[i]) == text) {
        return static_cast<int>(i);
      }
    }
    if (text.size() != 4) {
      break;
    }
    text += 'q'; // "e7e8" means promote to a queen
  }
  return -1;
}

// The node reached by playing `move`. The tree is kept as a chain of nodes so
// the network sees the game history, exactly as in self-play.
static Node *advance(Node *root, const Midnight::Move &move) {
  Midnight::Position next(root->position);
  play(next, move);

  for (Node *child : root->children) {
    if (child->position.hash() == next.hash()) {
      return child;
    }
  }
  return new Node(root, {}, next);
}

// The move that leads from `from` to its child `to`.
static std::string moveBetween(Node *from, Node *to) {
  for (const Midnight::Move &move : legalMoves(from->position)) {
    Midnight::Position next(from->position);
    play(next, move);
    if (next.hash() == to->position.hash()) {
      return uci(move);
    }
  }
  return "?";
}

static void announceResult(Node *root, Midnight::Color human) {
  const float value = terminalValue(root->position); // side to move's view
  if (value < 0.0f) {
    const bool humanMated = root->position.turn() == human;
    std::cout << "Checkmate. "
              << (humanMated ? "The engine wins." : "You win.") << std::endl;
  } else {
    std::cout << "Draw." << std::endl;
  }
}

static std::string trim(std::string text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.pop_back();
  }
  size_t start = 0;
  while (start < text.size() &&
         std::isspace(static_cast<unsigned char>(text[start]))) {
    start++;
  }
  return text.substr(start);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " checkpoint.pt [white|black]"
              << std::endl;
    return 1;
  }
  const std::string path = argv[1];
  const std::string side = argc > 2 ? argv[2] : "white";
  const Midnight::Color human =
      side == "black" ? Midnight::BLACK : Midnight::WHITE;

  try {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA, 0);
    }

    DNN model;
    torch::load(model, path, device);
    model->to(device);
    model->eval(); // batch norm / dropout must run in inference mode
    torch::NoGradGuard no_grad;

    GlobalData g = GlobalData(device);
    Node *root = new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));

    while (true) {
      std::cout << root->position << std::endl;

      if (isTerminal(root->position)) {
        announceResult(root, human);
        break;
      }

      if (root->position.turn() == human) {
        std::vector<Midnight::Move> moves = legalMoves(root->position);
        int index = -1;

        while (index < 0) {
          std::cout << "Your move: " << std::flush;
          std::string line;
          if (!std::getline(std::cin, line)) {
            return 0;
          }
          line = trim(line);

          if (line == "quit") {
            return 0;
          }
          if (line == "moves") {
            for (const Midnight::Move &move : moves) {
              std::cout << uci(move) << ' ';
            }
            std::cout << std::endl;
            continue;
          }
          if (line == "fen") {
            std::cout << root->position.fen() << std::endl;
            continue;
          }

          index = findMove(moves, line);
          if (index < 0) {
            std::cout << "Illegal or unreadable move. Type \"moves\" to list "
                         "the legal moves."
                      << std::endl;
          }
        }

        root = advance(root, moves[static_cast<size_t>(index)]);
      } else {
        std::cout << "Engine is thinking..." << std::endl;

        // Temperature 0: play the most visited move.
        Node *selected = getNextMove(root, model, 0.0f, g);
        if (selected == nullptr) {
          std::cout << "The engine has no move." << std::endl;
          break;
        }

        std::cout << "Engine plays " << moveBetween(root, selected);
        if (selected->visitCount > 0) {
          // Assumes the statistics convention of mcts.cpp: a node stores
          // values from its own side to move's view, so negate for the mover.
          const float eval = -(selected->totalValue / selected->visitCount);
          std::cout << " (eval " << eval << " for the engine, "
                    << selected->visitCount << " visits)";
        }
        std::cout << std::endl;

        root = selected;
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}
