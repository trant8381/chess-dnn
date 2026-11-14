// CPU createState (63-plane encoder) spot checks: piece planes must equal
// the bitboards, repetition/castling/turn/clocks planes the FEN state.
#include "test_util.h"

using namespace Midnight;

int main() {
  const torch::Device dev(torch::kCPU);

  // 1. START_FEN: 32 pieces, all four castling rights, white to move,
  //    zero clocks, no repetition.
  {
    Node *root = testutil::makeNode(START_FEN);
    auto hist = constructHistory(root);
    torch::Tensor planes = createState(hist, dev);
    REQUIRE_EQ(planes.size(0), 63L);
    auto a = planes.accessor<float, 3>();

    long material = 0;
    for (int c = 0; c < 2; c++) {
      for (int t = 0; t < 6; t++) {
        uint64_t bb = hist[0].pieces[t + c * 8];
        long ones = 0;
        for (int r = 0; r < 8; r++) {
          for (int f = 0; f < 8; f++) {
            const float expect = ((bb >> (r * 8 + f)) & 1ULL) ? 1.0f : 0.0f;
            REQUIRE(a[c * 6 + t][r][f] == expect);
            ones += (expect == 1.0f);
          }
        }
        material += ones;
      }
    }
    REQUIRE_EQ(material, 32L);

    // board-0 repetition planes are empty for a fresh position
    REQUIRE(planes[12].sum().item<float>() == 0.0f);
    REQUIRE(planes[13].sum().item<float>() == 0.0f);
    // turn plane (56): white -> zeros
    REQUIRE(planes[56].sum().item<float>() == 0.0f);
    // move counter (57) and fifty-move (62): zero
    REQUIRE(planes[57].sum().item<float>() == 0.0f);
    REQUIRE(planes[62].sum().item<float>() == 0.0f);
    // castling planes 58..61 all ones
    for (int p = 58; p <= 61; p++) {
      REQUIRE(planes[p].sum().item<float>() == 64.0f);
    }
    delete root;
  }

  // 2. Black to move, counters set, castling intact.
  {
    Node *root = testutil::makeNode("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 6 4");
    auto hist = constructHistory(root);
    torch::Tensor planes = createState(hist, dev);

    REQUIRE(planes[56].sum().item<float>() == 64.0f);  // black to move -> ones
    // move counter plane mirrors the position's own move counter
    const float movesVal = static_cast<float>(hist[0].moves()) / 100.0f;
    REQUIRE(std::fabs(planes[57].sum().item<float>() - 64.0f * movesVal) < 1e-5);
    // fifty-move plane mirrors the position's fifty-move counter
    const float fiftyVal = static_cast<float>(hist[0].fifty_move_rule()) / 100.0f;
    REQUIRE(std::fabs(planes[62].sum().item<float>() - 64.0f * fiftyVal) < 1e-5);
    for (int p = 58; p <= 61; p++) {
      REQUIRE(planes[p].sum().item<float>() == 64.0f);
    }
    delete root;
  }

  // 3. Knight dance repeated twice: the position recurs at ply 4 (index 4),
  //    inside the fifty-move repetition window -> TWO_FOLD plane lit on
  //    board 0 (the node position carries the game history).
  {
    Position pos(START_FEN);
    pos.play<WHITE>(Move(g1, f3));
    pos.play<BLACK>(Move(g8, f6));
    pos.play<WHITE>(Move(f3, g1));
    pos.play<BLACK>(Move(f6, g8));
    pos.play<WHITE>(Move(g1, f3));
    pos.play<BLACK>(Move(g8, f6));
    pos.play<WHITE>(Move(f3, g1));
    pos.play<BLACK>(Move(f6, g8));
    REQUIRE(pos.has_repetition(Position::TWO_FOLD));
    REQUIRE(pos.has_repetition(Position::THREE_FOLD) == false);
    Node *root = new Node(nullptr, {}, pos);
    auto hist = constructHistory(root);
    torch::Tensor planes = createState(hist, dev);
    REQUIRE(planes[12].sum().item<float>() == 64.0f);
    REQUIRE(planes[13].sum().item<float>() == 0.0f);
    delete root;
  }

  // 4. History depth: a 2-node chain -> board 1 mirrors the parent
  //    position, boards 2..3 fall back to START_FEN.
  {
    const std::string fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 6 4";
    Node *leaf = testutil::makeChain(fen, 2);
    auto hist = constructHistory(leaf);
    // board 1 mirrors the parent position; boards 2..3 fall back to START
    REQUIRE(hist[1].hash() == Midnight::Position(fen).hash());
    REQUIRE(hist[2].hash() == Midnight::Position(Midnight::START_FEN).hash());
    // board-1 12 piece planes == 2 rooks + king per side (6 pieces)
    torch::Tensor planes = createState(hist, dev);
    long b1 = 0;
    for (int c = 0; c < 2; c++) {
      for (int t = 0; t < 6; t++) {
        b1 += static_cast<long>(planes[14 + c * 6 + t].sum().item<float>());
      }
    }
    REQUIRE_EQ(b1, 6L);
    delete testutil::chainTop(leaf);
  }

  printf("test_state_cpu OK\n");
  return 0;
}
