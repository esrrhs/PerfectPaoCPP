// Unit tests for the Paodekuai engine: patterns, generator, must-beat,
// first-move-3, report-single, oracle, features, rule agent.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "algo/ppo.h"
#include "pdk/features.h"
#include "pdk/game.h"
#include "pdk/moves.h"
#include "pdk/oracle.h"
#include "pdk/rule_agent.h"
#include "torch_nn/models.h"

using namespace pdk;

#ifdef CHECK
#undef CHECK
#endif
#define CHECK(cond) ASSERT_TRUE(cond)

static void testDetect() {
    auto t = [](const std::string& s, int type) {
        MoveInfo i = detectMove(CardSet::parse(s));
        CHECK(i.type == type);
    };
    t("3", MT_SINGLE);
    t("2", MT_SINGLE);
    t("33", MT_PAIR);
    t("AA", MT_PAIR);
    t("333", MT_TRIPLE);
    t("3333", MT_BOMB);
    t("33345", MT_TRIPLE_TWO);   // two different kickers
    t("33344", MT_TRIPLE_TWO);   // pair kicker
    t("333KK", MT_TRIPLE_TWO);
    t("34567", MT_STRAIGHT);
    t("3456789TJQKA", MT_STRAIGHT);  // longest straight, ends at A
    CHECK(detectMove(CardSet::parse("3456789TJQKA2")).type == MT_WRONG);  // no 2
    CHECK(detectMove(CardSet::parse("23456")).type == MT_WRONG);
    t("3344", MT_PAIR_STRAIGHT);  // minimum pair chain of two
    t("33445566778899TTJJQQKKAA", MT_PAIR_STRAIGHT);  // len 10
    t("333444", MT_TRIPLE_STRAIGHT);
    t("333444555", MT_TRIPLE_STRAIGHT);
    t("33344456", MT_PLANE_SOLO);     // plane of 2 trios + 2 solo wings
    t("33344455", MT_PLANE_SOLO);     // two equal wings count as two solos
    t("3334445566", MT_PLANE_PAIR);   // plane of 2 trios + 2 pair wings
    // sub-chain interpretation: chain 345 + wings 3,4,5
    MoveInfo e = detectMove(CardSet::parse("333444555"));
    CHECK(e.type == MT_TRIPLE_STRAIGHT);
    MoveInfo e2 = detectMove(CardSet::parse("333444555345"));
    CHECK(e2.type == MT_PLANE_SOLO);
    CHECK(e2.rank == charRank('3'));
    CHECK(e2.len == 3);
    CHECK(detectMove(CardSet::parse("3333444555")).type == MT_WRONG);
    CHECK(detectMove(CardSet::parse("33445566")).type == MT_PAIR_STRAIGHT);
    CHECK(detectMove(CardSet::parse("3456")).type == MT_WRONG);   // straight < 5
    CHECK(detectMove(CardSet::parse("33445")).type == MT_WRONG);
    CHECK(detectMove(CardSet::parse("333444555666")).type == MT_TRIPLE_STRAIGHT);
    CHECK(detectMove(CardSet::parse("")).type == MT_PASS);
    // chains never include the 2
    CHECK(detectMove(CardSet::parse("3344556622")).type == MT_WRONG);
    CHECK(detectMove(CardSet::parse("TTTJJJQQQKKKAAA222")).type == MT_WRONG);
}

static void testGenerator() {
    CardSet hand = CardSet::parse("333344455667TA2");
    std::vector<CardSet> ms = genAllMoves(hand);
    std::set<uint64_t> keys;
    for (auto& m : ms) {
        CHECK(hand.contains(m));
        CHECK(!m.empty());
        CHECK(detectMove(m).type != MT_WRONG);
        keys.insert(m.key());
    }
    CHECK(keys.size() == ms.size());  // no duplicates
    // bombs present
    bool hasBomb = false;
    for (auto& m : ms)
        if (m == CardSet::parse("3333")) hasBomb = true;
    CHECK(hasBomb);

    // must-beat: only higher singles beat a 3 — and pass is NOT offered
    std::vector<CardSet> beat = legalMoves(hand, CardSet::parse("3"));
    bool hasSolo4 = false, hasSolo3 = false;
    for (auto& m : beat) {
        CHECK(!m.empty());  // pass illegal when a beat exists
        MoveInfo i = detectMove(m);
        CHECK(i.type == MT_SINGLE || i.type == MT_BOMB);
        if (m == CardSet::parse("4")) hasSolo4 = true;
        if (m == CardSet::parse("3")) hasSolo3 = true;
    }
    CHECK(hasSolo4 && !hasSolo3);

    // nothing beats: only pass
    std::vector<CardSet> none = legalMoves(CardSet::parse("35"), CardSet::parse("6"));
    CHECK(none.size() == 1);
    CHECK(none[0].empty());

    // bombs are mandatory beat candidates when they exist
    std::vector<CardSet> forced = legalMoves(CardSet::parse("33334"), CardSet::parse("3"));
    bool hasBomb2 = false, hasSingle4 = false, hasPass = false;
    for (auto& m : forced) {
        if (m.empty()) hasPass = true;
        if (m == CardSet::parse("3333")) hasBomb2 = true;
        if (m == CardSet::parse("4")) hasSingle4 = true;
    }
    CHECK(hasBomb2 && hasSingle4 && !hasPass);

    // only a bigger bomb beats a bomb
    std::vector<CardSet> vsBomb = legalMoves(CardSet::parse("33334444"),
                                             CardSet::parse("TTTT"));
    for (auto& m : vsBomb) {
        if (m.empty()) continue;
        CHECK(detectMove(m).type == MT_BOMB);
        CHECK(beats(m, CardSet::parse("TTTT")));
    }

    // free lead has no pass
    std::vector<CardSet> lead = legalMoves(hand, CardSet{});
    for (auto& m : lead) CHECK(!m.empty());

    // chains must have the same length to beat
    CardSet h2 = CardSet::parse("44455566678");
    std::vector<CardSet> bp = legalMoves(h2, CardSet::parse("333444"));
    bool hasLen2 = false;
    for (auto& m : bp) {
        if (m.empty()) continue;
        MoveInfo i = detectMove(m);
        if (i.type == MT_BOMB) continue;
        CHECK(i.len == 2);
        if (m == CardSet::parse("444555")) hasLen2 = true;
    }
    CHECK(hasLen2);

    // pair chain of length 2 beats pair chain of length 2 only
    std::vector<CardSet> pc = legalMoves(CardSet::parse("3344556677"),
                                         CardSet::parse("4455"));
    bool hasBeater = false;
    for (auto& m : pc) {
        if (m.empty()) continue;
        CHECK(beats(m, CardSet::parse("4455")));
        hasBeater = true;
    }
    CHECK(hasBeater);
}

static void testOracle() {
    auto steps = [](const std::string& s) {
        return minSteps(CardSet::parse(s));
    };
    CHECK(steps("34567") == 1);            // straight
    CHECK(steps("3344") == 1);             // minimum pair chain
    CHECK(steps("334455667788") == 1);     // long pair chain
    CHECK(steps("333444") == 1);           // bare triple chain
    CHECK(steps("33344456") == 1);         // plane + two solo wings
    CHECK(steps("33344455") == 1);         // plane + two equal wings
    CHECK(steps("3334445566") == 1);       // plane + two pair wings
    CHECK(steps("33334") == 2);            // bomb + single (no four-with-two)
    CHECK(steps("4") == 1);
    CHECK(steps("2") == 1);
    CHECK(steps("33345") == 1);            // triple + two kickers
    CHECK(steps("33344") == 1);            // triple + pair kickers
    CHECK(steps("3333") == 1);             // bomb
    CHECK(steps("333555") == 2);           // 333+55 then 5
    CHECK(steps("34") == 2);
    CHECK(steps("333444555") == 1);        // bare chain of 3 trios
    CHECK(steps("333444555666") <= 2);
    CHECK(steps("333444666777") == 2);     // two separated len-2 planes
    // random sanity over real dealt hands
    Rng rng(11);
    for (int t = 0; t < 100; ++t) {
        Game g;
        g.deal(rng);
        for (int s = 0; s < 3; ++s) {
            int v = minSteps(g.hand[s]);
            CHECK(v >= 1 && v <= kHandSize);
        }
    }
}

static void testDealAndFirstMove() {
    Rng rng(42);
    std::set<uint64_t> firstPlayers;
    for (int t = 0; t < 50; ++t) {
        Game g;
        g.deal(rng);
        for (int s = 0; s < 3; ++s) CHECK(g.hand[s].total() == kHandSize);
        // deck conservation: 3..K x4, A x3, 2 x1
        CardSet all;
        for (int s = 0; s < 3; ++s)
            for (int r = 0; r < kRanks; ++r) all.add(r, g.hand[s].c[r]);
        for (int r = 0; r <= 10; ++r) CHECK(all.c[r] == 4);
        CHECK(all.c[11] == 3);
        CHECK(all.c[12] == 1);
        // first player holds a 3 and every legal first move contains a 3
        CHECK(g.hand[g.turn].c[0] >= 1);
        std::vector<CardSet> legal = g.legal();
        CHECK(!legal.empty());
        for (const CardSet& m : legal) {
            CHECK(m.c[0] >= 1);
            CHECK(detectMove(m).type != MT_WRONG);
        }
        firstPlayers.insert(g.turn);
    }
    CHECK(firstPlayers.size() > 1);  // starter varies across deals
}

static void testReportSingle() {
    // seat 0 leads with hand 3,5; seat 1 holds a single card -> any solo lead
    // must be the largest solo (5)
    Game g;
    g.hand[0] = CardSet::parse("35");
    g.hand[1] = CardSet::parse("9");
    g.hand[2] = CardSet::parse("TTJQK");
    g.turn = 0;
    g.lastPlayer = 0;
    // pretend a full round already passed so the lead is free
    g.seq.push_back(CardSet::parse("K"));
    g.seq.push_back(CardSet{});  // pass
    g.seq.push_back(CardSet{});  // pass -> free lead again
    std::vector<CardSet> legal = g.legal();
    CHECK(!legal.empty());
    for (const CardSet& m : legal) {
        MoveInfo i = detectMove(m);
        CHECK(i.type != MT_SINGLE || m == CardSet::parse("5"));
    }
    bool has5 = false;
    for (const CardSet& m : legal)
        if (m == CardSet::parse("5")) has5 = true;
    CHECK(has5);

    // when the next player has NOT reported, small solos are fine
    Game g2;
    g2.hand[0] = CardSet::parse("35");
    g2.hand[1] = CardSet::parse("99");
    g2.hand[2] = CardSet::parse("TTJQK");
    g2.turn = 0;
    g2.lastPlayer = 0;
    g2.seq.push_back(CardSet::parse("K"));
    g2.seq.push_back(CardSet{});
    g2.seq.push_back(CardSet{});
    bool has3 = false;
    for (const CardSet& m : g2.legal())
        if (m == CardSet::parse("3")) has3 = true;
    CHECK(has3);
}

static void testRandomGames() {
    Rng rng(123);
    long long totalMoves = 0;
    int games = 300;
    std::array<int, 3> wins{};
    size_t maxOptions = 0;
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        int guard = 0;
        bool firstMove = true;
        while (!game.over) {
            std::vector<CardSet> legal = game.legal();
            CHECK(!legal.empty());
            // must-beat invariant
            bool hasPass = false, hasBeat = false;
            for (const CardSet& m : legal) {
                if (m.empty()) hasPass = true;
                else if (game.toBeat().empty() || beats(m, game.toBeat())) hasBeat = true;
            }
            if (game.toBeat().empty()) {
                CHECK(!hasPass);
            } else if (hasBeat) {
                CHECK(!hasPass);  // can beat => pass is illegal
            } else {
                CHECK(hasPass);   // cannot beat => only pass
            }
            maxOptions = std::max(maxOptions, legal.size());
            CardSet m = legal[rng.below(int(legal.size()))];
            CHECK(game.hand[game.turn].contains(m));
            if (firstMove) {
                CHECK(!m.empty());
                CHECK(m.c[0] >= 1);
                firstMove = false;
            }
            // every non-pass must beat the current rival play
            if (!m.empty() && !game.toBeat().empty()) {
                CHECK(beats(m, game.toBeat()));
            }
            game.step(m);
            ++totalMoves;
            ++guard;
            CHECK(guard < 10000);
        }
        ++wins[game.winner];
        CHECK(game.hand[game.winner].total() == 0);
        // zero-sum payoffs
        double p0 = game.payoff(0), p1 = game.payoff(1), p2 = game.payoff(2);
        CHECK(std::abs(p0 + p1 + p2) < 1e-9);
        CHECK(game.payoff(game.winner) > 0.0);
        for (int s = 0; s < 3; ++s)
            if (s != game.winner) CHECK(game.payoff(s) <= 0.0);
    }
    std::printf("  random games: %d, wins %d/%d/%d, avg moves %.1f, max options %zu\n",
                games, wins[0], wins[1], wins[2], double(totalMoves) / games,
                maxOptions);
    // seats symmetric: no seat should dominate random play
    for (int s = 0; s < 3; ++s) CHECK(wins[s] > 0);
    CHECK(maxOptions < size_t(torch_nn::kNumActions));
}

static void testShutdownDouble() {
    Game g;
    g.over = true;
    g.winner = 0;
    g.hand[1] = CardSet::parse("3456789TJQKA");  // 12 cards left
    g.hand[2] = CardSet{};                        // would be winner, fix below
    // make seat 2 the loser with a full unplayed hand
    g.hand[2].add(0, kHandSize);                  // impossible but only counts matter
    double p0 = g.payoff(0);
    double p1 = g.payoff(1);
    double p2 = g.payoff(2);
    CHECK(std::abs(p1 - (-12.0 / kHandSize)) < 1e-9);
    CHECK(std::abs(p2 - (-32.0 / kHandSize)) < 1e-9);  // full hand: doubled
    CHECK(std::abs(p0 - (32.0 + 12.0) / kHandSize) < 1e-9);
    CHECK(std::abs(p0 + p1 + p2) < 1e-9);
}

static void testFeatures() {
    Rng rng(9);
    for (int g = 0; g < 20; ++g) {
        Game game;
        game.deal(rng);
        int guard = 0;
        while (!game.over) {
            EncodedState e = encodeState(game);
            std::vector<LegalOption> opts = legalOptions(game);
            CHECK(!opts.empty());
            std::vector<CardSet> legal = game.legal();
            CHECK(opts.size() == legal.size());
            for (size_t i = 0; i < opts.size(); ++i) {
                const LegalOption& o = opts[i];
                CHECK(o.abstractId == static_cast<int>(i));
                CHECK(o.abstractId < torch_nn::kNumActions);
                CHECK(game.hand[game.turn].contains(o.concrete));
                if (!o.concrete.empty())
                    CHECK(detectMove(o.concrete).type != MT_WRONG);
                CHECK(o.feature[pdk::kActionSize - 1] == 1.0f);  // validity bit
                CHECK(o.feature[pdk::kCardMat + 0] ==
                      (detectMove(o.concrete).type == MT_BOMB ? 1.0f : 0.0f));
            }
            CardSet chosen = opts.front().concrete;
            bool inLegal = false;
            for (const CardSet& m : legal)
                if (m == chosen) inLegal = true;
            CHECK(inLegal);
            game.step(chosen);
            if (++guard > 10000) break;
        }
    }
}

static void testRuleAgent() {
    Rng rng(4242);
    int games = 200;
    std::array<int, 3> wins{};
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        int steps = 0;
        while (!game.over && steps < 10000) {
            int seat = game.turn;
            std::vector<CardSet> legal = game.legal();
            CHECK(!legal.empty());
            CardSet chosen = RuleAgent::selectMove(game, seat, legal);
            bool found = false;
            for (const CardSet& m : legal)
                if (m == chosen) found = true;
            CHECK(found);
            game.step(chosen);
            ++steps;
        }
        CHECK(game.over);
        ++wins[game.winner];
    }
    std::printf("  rule-agent games: %d, wins %d/%d/%d\n", games, wins[0], wins[1], wins[2]);
    for (int s = 0; s < 3; ++s) CHECK(wins[s] > 0);
}

// The rule agent must comfortably beat uniformly random play.
static void testRuleAgentBeatsRandom() {
    Rng rng(777);
    const int games = 300;
    int ruleWins = 0;
    for (int g = 0; g < games; ++g) {
        Game game;
        game.deal(rng);
        int ruleSeat = g % 3;
        while (!game.over) {
            int seat = game.turn;
            std::vector<CardSet> legal = game.legal();
            CardSet chosen = (seat == ruleSeat)
                                 ? RuleAgent::selectMove(game, seat, legal)
                                 : legal[rng.below(int(legal.size()))];
            game.step(chosen);
        }
        if (game.winner == ruleSeat) ++ruleWins;
    }
    double wp = double(ruleWins) / games;
    std::printf("  rule-agent vs random WP %.3f\n", wp);
    CHECK(wp > 0.45);
}

TEST(RulesTest, Detect) { testDetect(); }
TEST(RulesTest, Generator) { testGenerator(); }
TEST(RulesTest, Oracle) { testOracle(); }
TEST(RulesTest, DealAndFirstMove) { testDealAndFirstMove(); }
TEST(RulesTest, ReportSingle) { testReportSingle(); }
TEST(RulesTest, RandomGames) { testRandomGames(); }
TEST(RulesTest, ShutdownDouble) { testShutdownDouble(); }
TEST(RulesTest, Features) { testFeatures(); }
TEST(RulesTest, RuleAgent) { testRuleAgent(); }
TEST(RulesTest, RuleAgentBeatsRandom) { testRuleAgentBeatsRandom(); }
