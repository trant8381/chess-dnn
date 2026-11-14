// Move-generation tests for MidnightMoveGen: perft node counts on known
// positions, make/unmake round trips, fen parsing, stalemate / mate.
#include "test_util.h"
#include <cstdint>
#include <vector>

using namespace Midnight;

static std::vector<Move> legal(Position &pos) {
  if (pos.turn() == WHITE) {
    MoveList<WHITE> list(pos);
    return std::vector<Move>(list.begin(), list.end());
  }
  MoveList<BLACK> list(pos);
  return std::vector<Move>(list.begin(), list.end());
}

static uint64_t perft(Position &pos, int depth) {
  if (depth == 0) return 1;
  std::vector<Move> moves = legal(pos);
  uint64_t nodes = 0;
  for (const Move &mv : moves) {
    const Color mover = pos.turn();
    if (mover == WHITE) pos.play<WHITE>(mv);
    else pos.play<BLACK>(mv);
    nodes += perft(pos, depth - 1);
    if (mover == WHITE) pos.undo<WHITE>(mv);
    else pos.undo<BLACK>(mv);
  }
  return nodes;
}

static Square kingSquare(const Position &pos, Color c) {
  const Bitboard k = (c == WHITE) ? pos.occupancy<WHITE, KING>()
                                  : pos.occupancy<BLACK, KING>();
  REQUIRE(k != 0);
  return Square(__builtin_ctzll(k));
}

static bool inCheck(const Position &pos, Color c) {
  return pos.attackers_of(kingSquare(pos, c),
                          pos.occupancy<WHITE>() | pos.occupancy<BLACK>()) != 0;
}

int main() {
  // 1. Perft from the initial position (well-known node counts).
  {
    Position pos(START_FEN);
    REQUIRE_EQ(perft(pos, 1), 20ULL);
    Position pos2(START_FEN);
    REQUIRE_EQ(perft(pos2, 2), 400ULL);
    Position pos3(START_FEN);
    REQUIRE_EQ(perft(pos3, 3), 8902ULL);
    Position pos4(START_FEN);
    REQUIRE_EQ(perft(pos4, 4), 197281ULL);
  }

  // 2. Kiwipete (castling, pins, en-passant relevant).
  {
    Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    REQUIRE_EQ(perft(pos, 1), 48ULL);
    Position pos2("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    REQUIRE_EQ(perft(pos2, 2), 2039ULL);
    Position pos3("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    REQUIRE_EQ(perft(pos3, 3), 97862ULL);
  }

  // 3. Position 3 (pawns, knight forks, checks).
  {
    Position pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    REQUIRE_EQ(perft(pos, 1), 14ULL);
    Position pos2("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    REQUIRE_EQ(perft(pos2, 2), 191ULL);
    Position pos3("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    REQUIRE_EQ(perft(pos3, 3), 2812ULL);
    Position pos4("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    REQUIRE_EQ(perft(pos4, 4), 43238ULL);
  }

  // 4. Position 4 (promotions, many pins/checks).
  {
    Position pos("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    REQUIRE_EQ(perft(pos, 1), 6ULL);
    Position pos2("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    REQUIRE_EQ(perft(pos2, 2), 264ULL);
    Position pos3("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    REQUIRE_EQ(perft(pos3, 3), 9467ULL);
  }

  // 5. make/unmake restores the exact position (hash round trip).
  {
    Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const uint64_t before = pos.hash();
    const std::string fenBefore = pos.fen();
    std::vector<Move> moves = legal(pos);
    for (const Move &mv : moves) {
      const Color mover = pos.turn();
      if (mover == WHITE) pos.play<WHITE>(mv);
      else pos.play<BLACK>(mv);
      REQUIRE(pos.hash() != before);
      if (mover == WHITE) pos.undo<WHITE>(mv);
      else pos.undo<BLACK>(mv);
      REQUIRE(pos.hash() == before);
      REQUIRE(pos.fen() == fenBefore);
    }
  }

  // 6. fen round trip via the state hash.
  {
    const std::string fens[] = {
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 6 4",
    };
    for (const std::string &f : fens) {
      Position pos(f);
      Position again(pos.fen());
      REQUIRE(again.hash() == pos.hash());
      REQUIRE(again.turn() == pos.turn());
    }
  }

  // 7. Stalemate: zero legal moves, king not in check.
  {
    Position pos("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    REQUIRE(legal(pos).empty());
    REQUIRE(!inCheck(pos, BLACK));
  }

  // 8. Fool's mate built move-by-move: queen check is mate (0 moves, in
  // check), and the engine agrees the earlier setup lines are legal.
  {
    Position pos(START_FEN);
    REQUIRE_EQ(legal(pos).size(), 20ULL);
    pos.play<WHITE>(Move(f2, f3));
    pos.play<BLACK>(Move(e7, e5));
    pos.play<WHITE>(Move(g2, g4));
    pos.play<BLACK>(Move(d8, h4));
    REQUIRE(legal(pos).empty());
    REQUIRE(inCheck(pos, WHITE));
  }

  printf("test_movegen OK\n");
  return 0;
}
