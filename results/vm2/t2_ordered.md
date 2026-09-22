# The shared machine, run t2, re-scored under the ordering rule

Each array must hold at least as much as the one in front of it and be no quicker to
answer. Equal sizes allowed: the caches are non-inclusive, so a last level the size of
the level in front of it still adds capacity.

Stock 0.3832. Best legal design measured by any arm 0.4411 (+15.1%), found by the forest.

Share of the stock-to-best gap, by round:

| round | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| council | 17% | 17% | 29% | 48% | 56% | 68% | 70% | 72% | 72% | 72% | 72% | 72% |
| random  | 17% | 17% | 17% | 17% | 25% | 25% | 25% | 29% | 30% | 41% | 41% | 41% |
| forest  | 30% | 30% | 30% | 30% | 67% | 67% | 88% | 95% | 100% | 100% | 100% | 100% |

Finals: forest 0.4411 (+15.1%), council 0.4248 (+10.8%), random 0.4069 (+6.2%).

**Not a fair race.** No arm was searching under this rule; all three spent designs on
hierarchies now illegal, and the forest happened to land on a legal winner. What holds is
the council against random, handicapped alike: ahead at every round, 72% against 41%.

Unfiltered, the same run gave council 0.4580 (+19.5%) - a 1 MB first level at 6 cycles in
front of a 512 KB level at 4, which is what the rule now forbids.

Strict growth (each level must hold MORE) instead of at-least caps the arms at +9.6%
rather than +12.1%: 2.5 points rest on that one comparison.
