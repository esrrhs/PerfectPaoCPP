#include "pdk/features.h"

#include <algorithm>

#include "pdk/oracle.h"

namespace pdk {

namespace {

// row offsets inside the 11 x 13 card matrix
constexpr int R_COUNT0 = 0;  // count rows 0..3
constexpr int R_SOLO = 4;
constexpr int R_PAIR = 5;
constexpr int R_TRIO = 6;
constexpr int R_BOMB = 7;
constexpr int R_SOLOCHAIN = 8;
constexpr int R_PAIRCHAIN = 9;
constexpr int R_TRIOCHAIN = 10;

// Whether rank r sits inside a run of >= minLen consecutive ranks (3..A)
// each holding at least `need` copies.
bool inChain(const int8_t* cnt, int r, int minLen, int need) {
    if (r >= kChainRanks) return false;
    int lo = r, hi = r;
    while (lo - 1 >= 0 && cnt[lo - 1] >= need) --lo;
    while (hi + 1 < kChainRanks && cnt[hi + 1] >= need) ++hi;
    return hi - lo + 1 >= minLen;
}

}  // namespace

void cardMatrix(const CardSet& cs, uint8_t* out) {
    std::fill(out, out + kCardMat, 0);
    for (int r = 0; r < kRanks; ++r) {
        for (int k = 0; k < cs.c[r]; ++k)
            out[(R_COUNT0 + k) * kRanks + r] = 1;
        if (cs.c[r] >= 1) out[R_SOLO * kRanks + r] = 1;
        if (cs.c[r] >= 2) out[R_PAIR * kRanks + r] = 1;
        if (cs.c[r] >= 3) out[R_TRIO * kRanks + r] = 1;
        if (cs.c[r] == 4) out[R_BOMB * kRanks + r] = 1;
    }
    for (int r = 0; r < kChainRanks; ++r) {
        if (inChain(cs.c.data(), r, kMinStraight, 1))
            out[R_SOLOCHAIN * kRanks + r] = 1;
        if (inChain(cs.c.data(), r, kMinPairChain, 2))
            out[R_PAIRCHAIN * kRanks + r] = 1;
        if (inChain(cs.c.data(), r, kMinPlane, 3))
            out[R_TRIOCHAIN * kRanks + r] = 1;
    }
}

EncodedState encodeState(const Game& g) {
    EncodedState e;
    int seat = g.turn;
    int prev = g.prevSeat();
    int nxt = g.nextSeat();

    auto put = [&](int slot, const CardSet& cs) {
        cardMatrix(cs, e.imp.data() + slot * kCardMat);
    };
    CardSet others;
    for (int r = 0; r < kRanks; ++r)
        others.add(r, g.hand[prev].c[r] + g.hand[nxt].c[r]);
    put(0, g.hand[seat]);
    put(1, others);
    put(2, g.played[seat]);
    put(3, g.played[prev]);
    put(4, g.played[nxt]);
    put(5, g.lastMove[seat]);
    put(6, g.lastMove[prev]);
    put(7, g.lastMove[nxt]);

    // last 15 moves, chronological, zero-padded at the front
    int hlen = std::min<int>(kHistoryLen, static_cast<int>(g.seq.size()));
    for (int i = 0; i < hlen; ++i) {
        const CardSet& m = g.seq[g.seq.size() - hlen + i];
        put(kStaticMatrices + (kHistoryLen - hlen) + i, m);
    }

    auto oneHot = [](float* dst, int n, int value) {
        if (value >= 0 && value < n) dst[value] = 1.0f;
    };
    oneHot(e.scalar.data(), kHandHot, g.hand[prev].total() - 1);
    oneHot(e.scalar.data() + kHandHot, kHandHot, g.hand[nxt].total() - 1);
    oneHot(e.scalar.data() + 2 * kHandHot, kBombHot,
           std::min(g.bombCount, kBombHot - 1));
    e.scalar[2 * kHandHot + kBombHot + 0] = (g.lastPlayer == seat) ? 1.0f : 0.0f;
    e.scalar[2 * kHandHot + kBombHot + 1] = g.reported(nxt) ? 1.0f : 0.0f;
    e.scalar[2 * kHandHot + kBombHot + 2] = g.reported(prev) ? 1.0f : 0.0f;

    // perfect information extras
    cardMatrix(g.hand[prev], e.extra.data());
    cardMatrix(g.hand[nxt], e.extra.data() + kCardMat);
    e.extraScalar[0] = float(minSteps(g.hand[prev])) / float(kHandSize);
    e.extraScalar[1] = float(minSteps(g.hand[nxt])) / float(kHandSize);
    return e;
}

namespace {

// Whether no remaining card of the same pattern category can beat this move
// (bomb aside): the solo 2, the AA pair/triple, the A bomb, or a chain that
// tops out at A.
bool isLargest(const MoveInfo& a) {
    switch (a.type) {
        case MT_SINGLE: return a.rank == 12;  // the only 2
        case MT_BOMB:
        case MT_PAIR:
        case MT_TRIPLE:
        case MT_TRIPLE_TWO:
            return a.rank == 11;  // rank A (a pair of 2s cannot exist)
        case MT_STRAIGHT:
        case MT_PAIR_STRAIGHT:
        case MT_TRIPLE_STRAIGHT:
        case MT_PLANE_SOLO:
        case MT_PLANE_PAIR:
            return a.rank + a.len - 1 == 11;  // chain ends at A
        default:
            return false;
    }
}

}  // namespace

void legalOptions(const Game& g, std::vector<LegalOption>& out) {
    out.clear();
    std::vector<CardSet> concrete = g.legal();
    int seat = g.turn;
    int prev = g.prevSeat();
    int nxt = g.nextSeat();

    if (out.capacity() < concrete.size()) out.reserve(concrete.size());
    for (const CardSet& chosen : concrete) {
        MoveInfo info = detectMove(chosen);
        if (info.type == MT_WRONG && !chosen.empty()) continue;
        LegalOption o;
        o.abstractId = static_cast<int>(out.size());
        o.concrete = chosen;
        int size = chosen.total();
        std::array<uint8_t, kCardMat> card{};
        cardMatrix(chosen, card.data());
        for (int j = 0; j < kCardMat; ++j) o.feature[j] = float(card[j]);
        float* ex = o.feature.data() + kCardMat;
        ex[0] = (info.type == MT_BOMB) ? 1.0f : 0.0f;
        ex[1] = isLargest(info) ? 1.0f : 0.0f;
        ex[2] = (size > 0 && size == g.hand[nxt].total()) ? 1.0f : 0.0f;
        ex[3] = (size > 0 && size == g.hand[prev].total()) ? 1.0f : 0.0f;
        CardSet after = g.hand[seat];
        after.sub(chosen);
        ex[4] = float(minSteps(after)) / float(kHandSize);
        ex[5] = 1.0f;  // validity marker, needed for the all-zero pass matrix
        out.push_back(std::move(o));
    }
}

std::vector<LegalOption> legalOptions(const Game& g) {
    std::vector<LegalOption> out;
    legalOptions(g, out);
    return out;
}

}  // namespace pdk
