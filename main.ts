// Sokoban solver: push-based A* over rooms.
//
// Idea: in these levels every box stays inside its own room (a box pushed
// through a room's single door into the hallway can never come back), so a
// level splits into independent rooms, each with 3-4 boxes. Each room is
// solved push-optimally with A* over pushes, then the per-room move strings
// are stitched together with plain walks through the hallway. The search
// state is (sorted box cells, canonical player cell); the heuristic is the
// minimum-cost assignment of boxes to goals under box-free shortest-path
// distances (admissible: one push moves one box by one cell, so it lowers
// the assignment cost by at most one). Dead positions are pruned with
// dead-square analysis, 2x2 blocks and a freeze fixpoint. All budgets are
// counted in expanded states, never wall-clock time, so two runs of this
// program on the same input produce the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

// Deterministic work budgets (expanded states), never the clock.
const ROOM_BUDGET = 60000;
const WHOLE_BUDGET = 150000;

type Grid = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall (or outside the floor set of a subproblem)
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
};

function parse(text: string): Grid {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goals: number[] = [];
  const goalSet = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y] ?? '';
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet[i] = 1;
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// Breadth-first search over cells where pass(i) is true. Returns distance
// and parent arrays; parent[s] = -1. Also records the smallest reachable
// cell, which canonicalises the player's position inside its region.
function bfs(
  w: number,
  h: number,
  start: number,
  pass: (i: number) => boolean,
): { dist: Int32Array; parent: Int32Array; minCell: number } {
  const n = w * h;
  const dist = new Int32Array(n).fill(-1);
  const parent = new Int32Array(n).fill(-1);
  const queue = new Int32Array(n);
  let qh = 0;
  let qt = 0;
  dist[start] = 0;
  queue[qt++] = start;
  let minCell = start;
  while (qh < qt) {
    const c = queue[qh++];
    if (c < minCell) minCell = c;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const ni = ny * w + nx;
      if (dist[ni] >= 0 || !pass(ni)) continue;
      dist[ni] = dist[c]! + 1;
      parent[ni] = c;
      queue[qt++] = ni;
    }
  }
  return { dist, parent, minCell };
}

// Walk moves from `from` to `to` using a BFS parent array, or null.
function pathMoves(from: number, to: number, parent: Int32Array, w: number): string | null {
  if (from === to) return '';
  if (parent[to] === -1) return null;
  let out = '';
  let c = to;
  while (c !== from) {
    const p = parent[c]!;
    const d = c - p;
    out += d === -w ? 'u' : d === w ? 'd' : d === -1 ? 'l' : 'r';
    c = p;
  }
  return out.split('').reverse().join('');
}

// All permutations of [0..n), in lexicographic order (deterministic).
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a: number[] = [];
  for (let i = 0; i < n; i++) a.push(i);
  const gen = (k: number): void => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      const t = a[k]!;
      a[k] = a[i]!;
      a[i] = t;
      gen(k + 1);
      const t2 = a[k]!;
      a[k] = a[i]!;
      a[i] = t2;
    }
  };
  gen(0);
  return out;
}

// Cells from which a box can never reach a goal, by reverse pull analysis:
// a cell is live if it is a goal, or a box there could have been pushed
// from a live cell (the cell behind it must be floor for the player).
function deadCells(g: Grid, floor: Uint8Array): Uint8Array {
  const n = g.w * g.h;
  const live = new Uint8Array(n);
  const queue: number[] = [];
  for (const goal of g.goals) {
    if (floor[goal]) {
      live[goal] = 1;
      queue.push(goal);
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    const cx = c % g.w;
    const cy = (c / g.w) | 0;
    for (let d = 0; d < 4; d++) {
      // Box came from n = c - dir, pushed by a player standing at c - 2*dir.
      const nx = cx - DX[d]!;
      const ny = cy - DY[d]!;
      const bx = cx - 2 * DX[d]!;
      const by = cy - 2 * DY[d]!;
      if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
      if (bx < 0 || by < 0 || bx >= g.w || by >= g.h) continue;
      const ni = ny * g.w + nx;
      const bi = by * g.w + bx;
      if (!floor[ni] || live[ni] || floor[bi] === 0) continue;
      // The player cell must be floor (not a wall); it may be any floor cell.
      live[ni] = 1;
      queue.push(ni);
    }
  }
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (floor[i] && !live[i]) dead[i] = 1;
  return dead;
}

type PushResult = { moves: string; endPlayer: number; hasExit: boolean } | null;

// Push-optimal A* for one (sub)problem. `floor` marks the cells the player
// and boxes may occupy; `exitCell` (-1 when none) is a cell the player must
// still be able to reach once every box is on a goal, so the player is not
// left trapped inside a solved room. Prefers a solution with an exit; among
// those, the fewest pushes; a trapped optimal solution is returned only
// when no solution with an exit exists within two extra pushes.
function solvePush(
  g: Grid,
  floor: Uint8Array,
  startPlayer: number,
  boxes0: number[],
  budget: number,
  exitCell: number,
): PushResult {
  const n = g.w * g.h;
  const nb = boxes0.length;
  const goals = g.goals.filter((x) => floor[x]);
  if (goals.length !== nb) return null;
  const dead = deadCells(g, floor);
  for (const b of boxes0) if (dead[b]) return null;

  // Shortest-path distances from each goal, ignoring boxes.
  const goalDist: Int32Array[] = [];
  for (const goal of goals) {
    goalDist.push(bfs(g.w, g.h, goal, (i) => floor[i] === 1).dist);
  }
  const perms = nb <= 5 ? permutations(nb) : [];
  const heuristic = (boxes: number[]): number => {
    if (nb <= 5) {
      let best = Infinity;
      for (const p of perms) {
        let s = 0;
        for (let i = 0; i < nb; i++) {
          const d = goalDist[p[i]!]![boxes[i]!]!;
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let j = 0; j < nb; j++) {
        const d = goalDist[j]![b]!;
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };

  const isBlocked = (cell: number, boxSet: Uint8Array): boolean =>
    cell < 0 || cell >= n || floor[cell] === 0 || boxSet[cell] === 1;

  // A state is dead when a box sits on a dead cell, in a full 2x2 block of
  // walls/boxes without being on a goal, or frozen (it cannot move in any
  // direction and neither can the boxes blocking it).
  const isDead = (boxes: number[]): boolean => {
    for (const b of boxes) {
      if (dead[b]) return true;
      if (g.goalSet[b]) continue;
      const bx = b % g.w;
      const by = (b / g.w) | 0;
      // 2x2 blocks containing b: full of walls/boxes means b can never move.
      for (let oy = -1; oy <= 0; oy++) {
        for (let ox = -1; ox <= 0; ox++) {
          const x0 = bx + ox;
          const y0 = by + oy;
          if (x0 < 0 || y0 < 0 || x0 + 1 >= g.w || y0 + 1 >= g.h) continue;
          let full = true;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const c = (y0 + dy) * g.w + (x0 + dx);
              if (floor[c] === 0 || boxes.includes(c)) continue;
              full = false;
            }
          }
          if (full) return true;
        }
      }
    }
    // Freeze fixpoint: a box is frozen when every neighbour is a wall or a
    // frozen box. Only a frozen box off a goal kills the state.
    const frozen: number[] = [];
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of boxes) {
        if (frozen.includes(b)) continue;
        const bx = b % g.w;
        const by = (b / g.w) | 0;
        let allBlocked = true;
        for (let d = 0; d < 4; d++) {
          const nx = bx + DX[d]!;
          const ny = by + DY[d]!;
          if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
          const ni = ny * g.w + nx;
          if (floor[ni] === 0 || frozen.includes(ni)) continue;
          // Free floor, or a box that may still move: not blocked.
          allBlocked = false;
          break;
        }
        if (allBlocked) {
          frozen.push(b);
          changed = true;
        }
      }
    }
    for (const b of boxes) if (frozen.includes(b) && !g.goalSet[b]) return true;
    return false;
  };

  const canExit = (boxes: number[], player: number): boolean => {
    if (exitCell < 0) return true;
    if (floor[exitCell] === 0) return false;
    const boxSet = new Uint8Array(n);
    for (const b of boxes) boxSet[b] = 1;
    if (boxSet[exitCell]) return false;
    const { dist } = bfs(g.w, g.h, player, (i) => floor[i] === 1 && boxSet[i] === 0);
    return dist[exitCell]! >= 0;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => g.goalSet[b] === 1);

  // Node storage; a node's moves are the walk plus the push from its parent.
  const nodeBoxes: number[][] = [];
  const nodePlayer: number[] = [];
  const nodeG: number[] = [];
  const nodeParent: number[] = [];
  const nodeMoves: string[] = [];

  const startBoxes = boxes0.slice().sort((a, b) => a - b);
  const startReach = bfs(g.w, g.h, startPlayer, (i) => {
    if (floor[i] === 0) return false;
    for (const b of startBoxes) if (b === i) return false;
    return true;
  });
  const startCanon = startReach.minCell;
  const keyOf = (boxes: number[], canon: number): string => boxes.join(',') + '|' + canon;

  nodeBoxes.push(startBoxes);
  // The node keeps the player's REAL cell (walks start from it); only the
  // dedup key uses the canonical cell of the player's reachable region.
  nodePlayer.push(startPlayer);
  nodeG.push(0);
  nodeParent.push(-1);
  nodeMoves.push('');
  const best = new Map<string, number>();
  best.set(keyOf(startBoxes, startCanon), 0);

  // Binary heap on (f, -g, insertion order): smallest f, then deepest, then
  // earliest inserted, so ties break the same way every run.
  type HeapEntry = [number, number, number, number]; // f, g, order, node
  const heap: HeapEntry[] = [];
  let order = 0;
  const hLess = (a: HeapEntry, b: HeapEntry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!hLess(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && hLess(heap[c + 1]!, heap[c]!)) c++;
        if (!hLess(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return null;
  heapPush([h0, 0, order++, 0]);

  let bestGoal: { g: number; node: number } | null = null;
  let bestExit: { g: number; node: number } | null = null;
  let expanded = 0;

  const reconstruct = (node: number): { moves: string; endPlayer: number } => {
    let moves = '';
    let t = node;
    while (t > 0) {
      moves = nodeMoves[t]! + moves;
      t = nodeParent[t]!;
    }
    return { moves, endPlayer: nodePlayer[node]! };
  };

  while (heap.length > 0 && expanded < budget) {
    const [f, , , s] = heapPop();
    if (bestExit && f > bestExit.g) break;
    if (!bestExit && bestGoal && f > bestGoal.g + 2) break;
    const boxes = nodeBoxes[s]!;
    const playerPos = nodePlayer[s]!;
    const { dist, parent, minCell } = bfs(g.w, g.h, playerPos, (i) => {
      if (floor[i] === 0) return false;
      for (const b of boxes) if (b === i) return false;
      return true;
    });
    if (best.get(keyOf(boxes, minCell)) !== s) continue;
    if (solved(boxes)) {
      if (!bestGoal) bestGoal = { g: nodeG[s]!, node: s };
      if (canExit(boxes, playerPos) && !bestExit) {
        bestExit = { g: nodeG[s]!, node: s };
        if (bestGoal && bestExit.g === bestGoal.g) break;
      }
      continue;
    }
    expanded++;
    // The player stands on `playerPos`; reachable cells and walk paths.
    const boxSet = new Uint8Array(n);
    for (const b of boxes) boxSet[b] = 1;
    for (let bi = 0; bi < nb; bi++) {
      const b = boxes[bi]!;
      const bx = b % g.w;
      const by = (b / g.w) | 0;
      for (let d = 0; d < 4; d++) {
        const tx = bx + DX[d]!;
        const ty = by + DY[d]!;
        if (tx < 0 || ty < 0 || tx >= g.w || ty >= g.h) continue;
        const t = ty * g.w + tx;
        if (isBlocked(t, boxSet)) continue;
        const fx = bx - DX[d]!;
        const fy = by - DY[d]!;
        if (fx < 0 || fy < 0 || fx >= g.w || fy >= g.h) continue;
        const from = fy * g.w + fx;
        if (floor[from] === 0 || dist[from]! < 0) continue;
        const walk = pathMoves(playerPos, from, parent, g.w);
        if (walk === null) continue;
        const nb2 = boxes.slice();
        nb2[bi] = t;
        nb2.sort((a, b2) => a - b2);
        if (isDead(nb2)) continue;
        const hh = heuristic(nb2);
        if (hh === Infinity) continue;
        // Canonical player after the push: the smallest cell reachable
        // from the box's old cell with the new boxes in place. The node
        // itself keeps the real cell (the box's old cell).
        const boxSet2 = new Uint8Array(n);
        for (const x of nb2) boxSet2[x] = 1;
        const reach2 = bfs(g.w, g.h, b, (i) => floor[i] === 1 && boxSet2[i] === 0);
        const canon2 = reach2.minCell;
        const k = keyOf(nb2, canon2);
        const g2 = nodeG[s]! + 1;
        const prev = best.get(k);
        if (prev !== undefined && nodeG[prev]! <= g2) continue;
        const t2 = nodeBoxes.length;
        nodeBoxes.push(nb2);
        nodePlayer.push(b);
        nodeG.push(g2);
        nodeParent.push(s);
        nodeMoves.push(walk + CH[d]!);
        best.set(k, t2);
        heapPush([g2 + hh, g2, order++, t2]);
      }
    }
  }

  const pick = bestExit ?? bestGoal;
  if (!pick) return null;
  const r = reconstruct(pick.node);
  return { moves: r.moves, endPlayer: r.endPlayer, hasExit: bestExit !== null };
}

type Room = {
  cells: Uint8Array;
  entry: number; // room cell the player steps onto through the door
  doorOut: number; // hallway cell outside the door
  boxes: number[];
  goals: number[];
};

// Split a level into rooms by cutting one line (the player's row, or else
// the player's column): the hallway. Each remaining floor component holding
// boxes must hold as many goals, and touch the hallway through one door.
function decompose(g: Grid, horizontal: boolean): Room[] | null {
  const n = g.w * g.h;
  const pLine = horizontal ? (g.player / g.w) | 0 : g.player % g.w;
  const inLine = (i: number): boolean =>
    horizontal ? ((i / g.w) | 0) === pLine : i % g.w === pLine;
  const hallway = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (g.wall[i] === 0 && inLine(i)) hallway[i] = 1;
  const seen = new Uint8Array(n);
  const rooms: Room[] = [];
  const placedBox = new Uint8Array(n);
  const placedGoal = new Uint8Array(n);
  for (let s = 0; s < n; s++) {
    if (g.wall[s] || inLine(s) || seen[s]) continue;
    // Flood-fill one component off the hallway.
    const cells = new Uint8Array(n);
    const stack = [s];
    seen[s] = 1;
    cells[s] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      const cx = c % g.w;
      const cy = (c / g.w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        const ni = ny * g.w + nx;
        if (g.wall[ni] || inLine(ni) || seen[ni]) continue;
        seen[ni] = 1;
        cells[ni] = 1;
        stack.push(ni);
      }
    }
    const boxes = g.boxes.filter((b) => cells[b] === 1);
    const goals = g.goals.filter((x) => cells[x] === 1);
    if (boxes.length === 0 && goals.length === 0) continue; // empty pocket
    if (boxes.length === 0 || boxes.length !== goals.length) return null;
    // Door: component cells adjacent to a hallway floor cell.
    let entry = -1;
    let doorOut = -1;
    for (let i = 0; i < n; i++) {
      if (!cells[i]) continue;
      const cx = i % g.w;
      const cy = (i / g.w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        const ni = ny * g.w + nx;
        if (hallway[ni]) {
          if (entry === -1) {
            entry = i;
            doorOut = ni;
          } else if (entry !== i) {
            return null; // more than one door: not the shape expected
          }
        }
      }
    }
    if (entry === -1) return null;
    for (const b of boxes) placedBox[b] = 1;
    for (const x of goals) placedGoal[x] = 1;
    rooms.push({ cells, entry, doorOut, boxes, goals });
  }
  for (const b of g.boxes) if (!placedBox[b]) return null;
  for (const x of g.goals) if (!placedGoal[x]) return null;
  if (rooms.length === 0) return null;
  return rooms;
}

// Walk from `from` to `to` without pushing: boxes are obstacles. The walk
// stays on the hallway plus, when leaving a room, that room's cells.
function walkTo(
  g: Grid,
  from: number,
  to: number,
  boxes: number[],
  allowed: Uint8Array,
): string | null {
  const n = g.w * g.h;
  const boxSet = new Uint8Array(n);
  for (const b of boxes) boxSet[b] = 1;
  if (boxSet[to] || boxSet[from]) return null;
  const { parent } = bfs(g.w, g.h, from, (i) => allowed[i] === 1 && boxSet[i] === 0);
  return pathMoves(from, to, parent, g.w);
}

// Replay moves on the level; true only when every step is legal and every
// box ends on a goal.
function verify(g: Grid, moves: string): boolean {
  const n = g.w * g.h;
  const boxSet = new Uint8Array(n);
  for (const b of g.boxes) boxSet[b] = 1;
  let p = g.player;
  for (const ch of moves) {
    const d = ch === 'u' ? 0 : ch === 'd' ? 1 : ch === 'l' ? 2 : ch === 'r' ? 3 : -1;
    if (d < 0) return false;
    const px = p % g.w;
    const py = (p / g.w) | 0;
    const nx = px + DX[d]!;
    const ny = py + DY[d]!;
    if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) return false;
    const ni = ny * g.w + nx;
    if (g.wall[ni]) return false;
    if (boxSet[ni]) {
      const bx = nx + DX[d]!;
      const by = ny + DY[d]!;
      if (bx < 0 || by < 0 || bx >= g.w || by >= g.h) return false;
      const bi = by * g.w + bx;
      if (g.wall[bi] || boxSet[bi]) return false;
      boxSet[ni] = 0;
      boxSet[bi] = 1;
    }
    p = ni;
  }
  return g.boxes.every((b) => boxSet[b] === 0 || g.goalSet[b] === 1) &&
    (() => {
      for (let i = 0; i < n; i++) if (boxSet[i] && !g.goalSet[i]) return false;
      return true;
    })();
}

function solveLevel(text: string): string {
  const g = parse(text);
  // 1. Room-by-room: solve each room from its door, then stitch.
  for (const horizontal of [true, false]) {
    const rooms = decompose(g, horizontal);
    if (!rooms) continue;
    const n = g.w * g.h;
    const hallway = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (g.wall[i] === 0) {
        const onLine = horizontal
          ? ((i / g.w) | 0) === ((g.player / g.w) | 0)
          : i % g.w === g.player % g.w;
        if (onLine) hallway[i] = 1;
      }
    }
    // Solve every room off-line, from its door.
    type SolvedRoom = { room: Room; res: NonNullable<PushResult> };
    const solvedRooms: SolvedRoom[] = [];
    let allSolved = true;
    for (const room of rooms) {
      const floor = room.cells.slice();
      floor[room.doorOut] = 1;
      const res = solvePush(g, floor, room.doorOut, room.boxes, ROOM_BUDGET, room.doorOut);
      if (!res) {
        allSolved = false;
        break;
      }
      solvedRooms.push({ room, res });
    }
    if (!allSolved) continue;
    // Rooms the player can leave go first; a room that traps the player
    // (its solution leaves no way out) goes last.
    solvedRooms.sort((a, b) => Number(b.res.hasExit) - Number(a.res.hasExit));
    // Stitch: walk to each door, run the room's moves, walk on.
    let moves = '';
    let player = g.player;
    let curBoxes = g.boxes.slice();
    let curRoom: Room | null = null;
    let ok = true;
    for (const { room, res } of solvedRooms) {
      const allowed = hallway.slice();
      if (curRoom) {
        for (let i = 0; i < n; i++) if (curRoom.cells[i]) allowed[i] = 1;
        allowed[curRoom.doorOut] = 1;
      }
      const walk = walkTo(g, player, room.doorOut, curBoxes, allowed);
      if (walk === null) {
        ok = false;
        break;
      }
      moves += walk;
      // Apply the walk, then the room's moves, to the tracked state.
      const combined = walk + res.moves;
      for (const ch of combined) {
        const d = ch === 'u' ? 0 : ch === 'd' ? 1 : ch === 'l' ? 2 : 3;
        const px = player % g.w;
        const py = (player / g.w) | 0;
        const ni = (py + DY[d]!) * g.w + (px + DX[d]!);
        const at = curBoxes.indexOf(ni);
        if (at >= 0) curBoxes[at] = (py + 2 * DY[d]!) * g.w + (px + 2 * DX[d]!);
        player = ni;
      }
      moves += res.moves;
      curRoom = room;
    }
    if (ok && verify(g, moves)) return moves;
  }
  // 2. Fallback: one push-A* over the whole level.
  const floorAll = new Uint8Array(g.w * g.h);
  for (let i = 0; i < g.w * g.h; i++) if (g.wall[i] === 0) floorAll[i] = 1;
  const res = solvePush(g, floorAll, g.player, g.boxes, WHOLE_BUDGET, -1);
  if (res && verify(g, res.moves)) return res.moves;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
