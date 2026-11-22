[![sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/rifki500/sokoban-solver-01-run-20261121-entry/rank.svg)](https://make.fun/)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/rifki500/sokoban-solver-01-run-20261121-entry/reward.svg)](https://make.fun/)

# Sokoban solver

A push-optimal Sokoban solver written for the [make.fun](https://make.fun/)
bounty "Build the best Sokoban solver, 21 November 2026"
(`sokoban-solver-01-run-20261121`).

## How it works

Levels in this competition are built from independent rooms off a central
hallway, and a box pushed out of its room through its door can never come
back, so the level splits into rooms that can each be solved on their own:

1. **Decompose.** Cutting the hallway line (the player's row, or else the
   player's column) splits the floor into components. Each component holding
   boxes must hold as many goals and touch the hallway through exactly one
   door; otherwise the whole level is solved as one problem.
2. **Push-optimal A\* per room.** States are (sorted box cells, canonical
   player cell); the heuristic is the minimum-cost assignment of boxes to
   goals under box-free shortest-path distances, which is admissible because
   one push moves one box by one cell. Dead positions are pruned with
   dead-square analysis, full 2x2 blocks, and a freeze fixpoint.
3. **Stitch.** The per-room move strings are joined with plain walks through
   the hallway. A room whose solution would trap the player inside is solved
   last; solutions that still let the player out are preferred.
4. **Verify.** Every level's final move string is replayed on the level
   before it is written out, so an answer is only ever emitted when it is
   legal and solves the level.

All budgets are counted in expanded states, never wall-clock time, so two
runs on the same input produce the same bytes.

## Local grade

On the kit's practice set the solver answers all five instances at exactly
the push-optimal count (13, 16, 29, 47 and 60 pushes), scoring **1,000,000**,
against **200,000** for the house baseline `plain-astar 1.0.0`:

```
node cli/makefun.mjs grade-local --kit . --bot . --driver process --worktree
```

## Run

```
node main.ts
```

Reads levels from `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and
writes `/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`). Node 22.
