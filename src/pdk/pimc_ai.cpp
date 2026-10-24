#include "pdk/pimc_ai.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "pdk/oracle.h"
#include "pdk/rule_agent.h"

namespace pdk {

namespace {

// ---------------------------------------------------------------------------
// Fully observable simulation state used inside one determinized world.
// ---------------------------------------------------------------------------
struct SimState {
    std::array<CardSet, 3> hand;
    CardSet table;    // move to beat; empty = free lead
    int lastPlayer = 0;
    int turn = 0;
    int passes = 0;   // consecutive passes since the last non-pass move
};

// Same legality semantics as Game::legal() minus the first-move-3 rule
// (the search never sits at the very first decision of a game): free lead has
// no pass; must-beat when following; report-single solo topping on lead.
std::vector<CardSet> simLegal(const SimState& s) {
    if (s.table.empty()) {
        std::vector<CardSet> ms = genAllMoves(s.hand[s.turn]);
        int nxt = (s.turn + 1) % 3;
        if (s.hand[nxt].total() == 1) {
            int largest = -1;
            for (int r = 0; r < kRanks; ++r)
                if (s.hand[s.turn].c[r] >= 1) largest = r;
            size_t w = 0;
            for (auto& m : ms) {
                MoveInfo info = detectMove(m);
                if (info.type != MT_SINGLE || info.rank == largest)
                    ms[w++] = std::move(m);
            }
            ms.resize(w);
        }
        return ms;
    }
    return legalMoves(s.hand[s.turn], s.table);
}

void simApply(SimState& s, const CardSet& m) {
    if (!m.empty()) {
        s.hand[s.turn].sub(m);
        s.table = m;
        s.lastPlayer = s.turn;
        s.passes = 0;
    } else {
        if (++s.passes >= 2) {
            s.table.clear();  // everyone passed: the last player leads again
            s.passes = 0;
        }
    }
    s.turn = (s.turn + 1) % 3;
}

int simWinner(const SimState& s) {
    for (int p = 0; p < 3; ++p)
        if (s.hand[p].empty()) return p;
    return -1;
}

// Terminal utilities in [0, 1]: winner 0.6 + share of the loot, losers scaled
// by how few cards they are holding.
std::array<double, 3> simUtilities(const SimState& s, int winner, Rng& rng) {
    std::array<double, 3> u{};
    double loot = 0.0;
    for (int p = 0; p < 3; ++p)
        if (p != winner) loot += double(s.hand[p].total()) / kHandSize;
    for (int p = 0; p < 3; ++p)
        u[p] = (p == winner) ? (0.6 + 0.4 * loot / 2.0)
                             : (0.4 * (1.0 - double(s.hand[p].total()) / kHandSize));
    (void)rng;
    return u;
}

// Depth-capped evaluation when a rollout drags on: fewest cards wins, ties by
// a second criterion, still deterministic.
std::array<double, 3> simEvalCapped(const SimState& s) {
    int best = 0;
    for (int p = 1; p < 3; ++p)
        if (s.hand[p].total() < s.hand[best].total()) best = p;
    std::array<double, 3> u{};
    for (int p = 0; p < 3; ++p)
        u[p] = (p == best) ? 0.7 : 0.15;
    return u;
}

// Rollout policy: a lightweight Oracle-planning chooser (min-steps on the
// resulting hand with cheap tactical guards). Memoized oracle makes this
// affordable and much stronger than pure greedy rollouts.
CardSet rolloutChoose(const SimState& s, const std::vector<CardSet>& legal) {
    if (legal.size() == 1) return legal.front();
    int seat = s.turn;
    int handTotal = s.hand[static_cast<size_t>(seat)].total();
    for (const CardSet& m : legal) {
        if (!m.empty() && m.total() == handTotal) return m;
    }
    bool leading = s.table.empty();
    CardSet best = legal.front();
    double bestScore = -1e18;
    for (const CardSet& m : legal) {
        if (m.empty()) continue;
        MoveInfo info = detectMove(m);
        CardSet after = s.hand[static_cast<size_t>(seat)];
        after.sub(m);
        double score = -double(minSteps(after)) * 10.0 - double(info.rank);
        if (leading) {
            score += 0.5 * double(m.total());
            if (info.type == MT_BOMB) score -= 6.0;
        } else {
            for (int rr = 0; rr < kRanks; ++rr) {
                if (s.hand[static_cast<size_t>(seat)].c[rr] == 4 &&
                    m.c[rr] > 0 && m.c[rr] < 4)
                    score -= 8.0;  // never break a bomb for a kicker
            }
            if (info.type == MT_BOMB) {
                int rivalHand = s.hand[static_cast<size_t>(s.lastPlayer)].total();
                if (rivalHand > 2 && minSteps(after) > 1) score -= 15.0;
            }
        }
        if (score > bestScore) {
            bestScore = score;
            best = m;
        }
    }
    return best;
}

// Planning rollout over sim legality.
std::array<double, 3> fastRollout(SimState s, int depthCap) {
    int winner = simWinner(s);
    int guard = 0;
    while (winner < 0 && guard < depthCap) {
        std::vector<CardSet> legal = simLegal(s);
        if (legal.empty()) break;  // defensive; engine guarantees a pass
        CardSet m = rolloutChoose(s, legal);
        simApply(s, m);
        winner = simWinner(s);
        ++guard;
    }
    Rng dummy(1);
    if (winner >= 0) return simUtilities(s, winner, dummy);
    return simEvalCapped(s);
}

// ---------------------------------------------------------------------------
// Max^N UCB MCTS over one determinized world.
// ---------------------------------------------------------------------------
struct Node {
    SimState st;
    std::vector<CardSet> moves;
    std::vector<int> child;      // arena index or -1
    int untried = 0;             // next move index to expand
    std::array<double, 3> util{};
    int visits = 0;
    int winner = -1;             // >= 0 when terminal
};

class Arena {
public:
    int newNode(SimState st) {
        nodes_.emplace_back();
        Node& n = nodes_.back();
        n.st = std::move(st);
        n.winner = simWinner(n.st);
        if (n.winner < 0) {
            n.moves = simLegal(n.st);
            n.child.assign(n.moves.size(), -1);
            n.untried = 0;
        }
        return static_cast<int>(nodes_.size()) - 1;
    }
    Node& operator[](int i) { return nodes_[static_cast<size_t>(i)]; }

private:
    std::vector<Node> nodes_;
};

void runMcts(Arena& arena, int rootIdx, const PimcConfig& cfg, Rng& rng) {
    std::vector<int> path;
    for (int it = 0; it < cfg.iterations; ++it) {
        path.clear();
        int cur = rootIdx;
        // Selection: descend while fully expanded and non-terminal.
        while (true) {
            Node& node = arena[cur];
            path.push_back(cur);
            if (node.winner >= 0 || node.moves.empty()) break;
            if (node.untried < static_cast<int>(node.moves.size())) {
                // Expansion: one new child per iteration. Copy what we need
                // first: Arena::newNode may reallocate and invalidate the
                // `node` reference above.
                int idx = node.untried;
                SimState ns = node.st;
                node.untried = idx + 1;
                simApply(ns, node.moves[static_cast<size_t>(idx)]);
                int childIdx = arena.newNode(std::move(ns));
                arena[cur].child[static_cast<size_t>(idx)] = childIdx;
                path.push_back(childIdx);
                cur = childIdx;
                break;
            }
            // UCB1 over children for the player to move at this node.
            int player = node.st.turn;
            int bestChild = -1;
            double bestValue = -1e18;
            double logN = std::log(double(node.visits + 1));
            for (size_t c = 0; c < node.moves.size(); ++c) {
                int ci = node.child[c];
                if (ci < 0) continue;
                Node& ch = arena[ci];
                double mean = ch.visits > 0 ? ch.util[static_cast<size_t>(player)] / ch.visits : 0.0;
                double explore = ch.visits > 0
                                     ? cfg.exploration * std::sqrt(logN / ch.visits)
                                     : 1e6;
                double v = mean + explore;
                if (v > bestValue) {
                    bestValue = v;
                    bestChild = ci;
                }
            }
            if (bestChild < 0) break;
            cur = bestChild;
        }

        // Simulation from the reached node.
        std::array<double, 3> u;
        Node& leaf = arena[cur];
        if (leaf.winner >= 0) {
            u = simUtilities(leaf.st, leaf.winner, rng);
        } else {
            u = fastRollout(leaf.st, cfg.rolloutDepth);
        }

        // Backpropagation along the path.
        for (int ni : path) {
            Node& n = arena[ni];
            for (int p = 0; p < 3; ++p) n.util[static_cast<size_t>(p)] += u[static_cast<size_t>(p)];
            n.visits += 1;
        }
    }
}

CardSet fullDeckSet() {
    CardSet d;
    for (int r = 0; r <= 10; ++r) d.add(r, 4);
    d.add(11, 3);
    d.add(12, 1);
    return d;
}

}  // namespace

std::vector<PimcAi::MoveStat> PimcAi::decide(const Game& g, int seat) {
    std::vector<MoveStat> out;
    std::vector<CardSet> candidates = g.legal();
    if (candidates.empty()) return out;
    if (candidates.size() == 1) {
        out.push_back({candidates.front(), 1, 1.0, 1});
        return out;
    }
    // finish-now shortcut
    for (const CardSet& m : candidates) {
        if (!m.empty() && m.total() == g.hand[seat].total()) {
            out.push_back({m, 1, 1.0, 1});
            return out;
        }
    }

    // Determinization pool: 48 cards minus my hand minus everything played.
    CardSet unseen = fullDeckSet();
    unseen.sub(g.hand[seat]);
    for (int p = 0; p < 3; ++p) unseen.sub(g.played[p]);
    std::vector<int> pool;
    for (int r = 0; r < kRanks; ++r)
        for (int k = 0; k < unseen.c[r]; ++k) pool.push_back(r);

    int o1 = (seat + 1) % 3;
    int o2 = (seat + 2) % 3;
    int n1 = g.hand[o1].total();
    int n2 = g.hand[o2].total();
    if (n1 + n2 != static_cast<int>(pool.size())) {
        // inconsistent view (defensive): fall back to greedy
        CardSet m = RuleAgent::selectMove(g, seat, candidates);
        out.push_back({m, 1, 1.0, 1});
        return out;
    }

    struct Stat {
        long long visits = 0;
        double utilSum = 0.0;
        int samples = 0;
    };
    std::unordered_map<uint64_t, Stat> stats;
    stats.reserve(candidates.size() * 2);

    for (int w = 0; w < cfg_.worlds; ++w) {
        // shuffle the pool, deal n1 / n2 to the two opponents
        for (size_t i = pool.size(); i > 1; --i) {
            size_t j = static_cast<size_t>(rng_.below(static_cast<int>(i)));
            std::swap(pool[i - 1], pool[j]);
        }
        SimState st;
        st.hand[seat] = g.hand[seat];
        st.hand[o1].clear();
        st.hand[o2].clear();
        for (int i = 0; i < n1; ++i) st.hand[o1].add(pool[static_cast<size_t>(i)]);
        for (int i = n1; i < n1 + n2; ++i) st.hand[o2].add(pool[static_cast<size_t>(i)]);
        st.table = g.toBeat();
        st.lastPlayer = g.lastPlayer;
        st.turn = seat;
        st.passes = 0;
        if (st.table.empty()) st.lastPlayer = seat;

        Arena arena;
        int root = arena.newNode(st);
        runMcts(arena, root, cfg_, rng_);

        Node& root_ = arena[root];
        for (size_t c = 0; c < root_.moves.size(); ++c) {
            int ci = root_.child[c];
            if (ci < 0) continue;
            Node& ch = arena[ci];
            if (ch.visits <= 0) continue;
            Stat& s = stats[root_.moves[c].key()];
            s.visits += ch.visits;
            s.utilSum += ch.util[static_cast<size_t>(seat)];
            s.samples += 1;
        }
    }

    for (const CardSet& m : candidates) {
        auto it = stats.find(m.key());
        if (it == stats.end() || it->second.visits == 0) {
            out.push_back({m, 0, 0.0, 0});
        } else {
            const Stat& s = it->second;
            out.push_back({m, s.visits, s.utilSum / double(s.visits), s.samples});
        }
    }
    std::sort(out.begin(), out.end(), [](const MoveStat& a, const MoveStat& b) {
        if (a.visits == 0 && b.visits == 0) return false;
        if (a.visits == 0) return false;
        if (b.visits == 0) return true;
        if (a.avgWinRate != b.avgWinRate) return a.avgWinRate > b.avgWinRate;
        return a.visits > b.visits;
    });
    if (out.empty() || out.front().visits == 0) {
        // zero-budget search: greedy ranking so the caller still gets an order
        CardSet m = RuleAgent::selectMove(g, seat, candidates);
        out.insert(out.begin(), {m, 1, 1.0, 1});
    }
    return out;
}

CardSet PimcAi::selectMove(const Game& g, int seat) {
    std::vector<MoveStat> ranked = decide(g, seat);
    if (ranked.empty()) return CardSet();
    return ranked.front().move;
}

}  // namespace pdk
