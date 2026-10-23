// Oracle: minimum number of moves to play out all cards of a Paodekuai hand
// (max 16 cards). Same structure as the PerfectDou oracle: a count-based DP
// over group multiplicities for all non-chain plays, combined with a memoized
// DFS enumerating straights / pair chains / planes (with wings).
//
// Used for:
//   - the shaped "distance to win" oracle reward
//   - perfect-information critic features and action features.
#pragma once

#include <cstddef>
#include <cstdint>

#include "pdk/cards.h"

namespace pdk {

// Minimum play-out steps for the hand. Results are memoized per thread.
int minSteps(const CardSet& hand);

// Clears the thread-local memoization table.
void clearOracleCache();

// Current cache size (number of memoized states).
size_t oracleCacheSize();

}  // namespace pdk
