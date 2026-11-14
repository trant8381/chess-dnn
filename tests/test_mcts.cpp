// End-to-end MCTS search through the shared Predictor: legal selections,
// real visit statistics, temperature modes, terminal handling, and a short
// pruned self-play progression. Runs on CPU and GPU (argv: cpu / gpu).
#include "test_util.h"

static bool isOneLegalMove(const Midnight::Position &rootPos,
                           const Midnight::Position &childPos) {
  std::vector<Midnight::Move> legal = createMovelistVec(rootPos);
  for (const Midnight::Move &mv : legal) {
    Midnight::Position copy = rootPos;
    if (copy.turn() == Midnight::WHITE) {
      copy.play<Midnight::WHITE>(mv);
    } else {
      copy.play<Midnight::BLACK>(mv);
    }
    if (copy.fen() == childPos.fen()) return true;
  }
  return false;
}

int main(int argc, char **argv) {
  torch::Device dev = testutil::pickDevice(argc, argv);
  printf("device: %s\n", dev.str().c_str());

  Predictor predictor(dev);
  GlobalData g(dev, &predictor);

  // 1. temperature 0: most-visited move, legal and evaluated
  {
    Node *root =
        new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
    Node *sel = getNextMove(root, 0.0f, g);
    REQUIRE(sel != nullptr);
    REQUIRE(sel->visitCount > 0);
    REQUIRE(root->visitCount > 0);
    REQUIRE(isOneLegalMove(root->position, sel->position));
    REQUIRE(sel->valueEval != INFINITY);
    delete root;  // frees the whole subtree incl. sel
  }

  // 2. temperature 1: stochastic draw still returns a legal, visited move
  {
    Node *root =
        new Node(nullptr, {}, Midnight::Position(Midnight::START_FEN));
    Node *sel = getNextMove(root, 1.0f, g);
    REQUIRE(sel != nullptr);
    REQUIRE(sel->visitCount > 0);
    REQUIRE(isOneLegalMove(root->position, sel->position));
    delete root;
  }

  // 3. terminal root (fool's mate) -> nullptr, no search attempted
  {
    Node *root = new Node(
        nullptr, {},
        Midnight::Position(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"));
    REQUIRE(getNextMove(root, 0.0f, g) == nullptr);
    delete root;
  }

  // 4. four-ply self-play with tree pruning between moves
  {
    Midnight::Position start(Midnight::START_FEN);
    Node *root = new Node(nullptr, {}, start);
    for (int ply = 0; ply < 4; ply++) {
      Node *sel = getNextMove(root, 1.0f, g);
      REQUIRE(sel != nullptr);
      REQUIRE(sel->visitCount > 0);
      REQUIRE(isOneLegalMove(root->position, sel->position));
      Node *old = root;
      old->children.erase(sel);
      root = sel;
      root->parent = nullptr;
      delete old;  // prunes the unselected subtrees
    }
    delete root;
  }

  printf("test_mcts OK\n");
  return 0;
}
