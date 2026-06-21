#!/usr/bin/env python3
"""
generate_stress_tests_training.py

Processes problems one at a time:
  1. Auto-generate a stress test generator from stdin format
  2. Run generator -> pipe to ref solution
  3. If pass: embed in cc_pool_cache.json immediately, move on
  4. If fail: retry once with same generator, then skip

Usage:
    python3 online_rl/generate_stress_tests_training.py [--limit N]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAIN_CACHE = "online_rl/cc_pool_cache.json"


# ── Manual overrides for tricky problems ─────────────────────────────────────
# If a task_id is here, this generator is used instead of the auto-generator.
MANUAL: dict[str, str] = {

"cc_80_C. Heroes": """\
import random, string
random.seed(42)
names = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Heidi",
         "Ivan","Judy","Mallory","Niaj","Olivia","Peggy","Rupert",
         "Sybil","Trent","Victor","Walter","Xavier","Yvonne","Zara",
         "Aaron","Bella","Chris","Diana","Edward","Fiona","George",
         "Hannah","Igor","Julia","Kevin","Laura","Mike","Nancy",
         "Oscar","Paula","Quinn","Rachel","Steve","Trudy","Uma",
         "Vera","Will","Xena","Yasmin","Zeke","Anna","Boris"]
n = 42
pairs = []
for i in range(n):
    a, b = names[i % len(names)], names[(i+7) % len(names)]
    if a != b:
        pairs.append((a, b))
pairs = pairs[:n]
print(len(pairs))
for a, b in pairs:
    print(f"{a} likes {b}")
""",

"cc_339_A. Helpful Maths": """\
import random
random.seed(42)
parts = [str(random.randint(1,3)) for _ in range(200)]
print('+'.join(parts))
""",

"cc_474_A. Keyboard": """\
import random, string
random.seed(42)
print(random.choice('LR'))
print(''.join(random.choices(string.ascii_lowercase + ',.', k=50000)))
""",

"cc_733_A. Grasshopper And the String": """\
import random, string
random.seed(42)
print(''.join(random.choices(string.ascii_uppercase, k=100)))
""",

"cc_802_G. Fake News (easy)": """\
import random, string
random.seed(42)
print(''.join(random.choices(string.ascii_lowercase, k=100)))
""",

"cc_712_B. Memory and Trident": """\
import random
random.seed(42)
n = 50000
dirs = list('LRUD')
print(''.join(random.choices(dirs, k=n)))
""",

"cc_521_A. DNA Alignment": """\
import random
random.seed(42)
n = 50000
print(n)
print(''.join(random.choices('ACGT', k=n)))
""",

"cc_1181_A. Chunga-Changa": """\
print(10**18, 10**18, 1)
""",

"cc_171_A. Mysterious numbers - 1": """\
print(10**9, 10**9)
""",

"cc_312_B. Archer": """\
import random
random.seed(42)
print(random.randint(1,10**9), random.randint(1,10**9),
      random.randint(1,10**9), random.randint(1,10**9))
""",

"cc_136_B. Ternary Logic": """\
print(10**9, 10**9-1)
""",

"cc_513_A. Game": """\
import random
random.seed(42)
print(*[random.randint(1,10**9) for _ in range(4)])
""",

"cc_215_A. Bicycle Chain": """\
import random
random.seed(42)
n, m = 50, 45
a = sorted(random.sample(range(1,5000), n))
b = sorted(random.sample(range(1,5000), m))
print(n)
print(*a)
print(m)
print(*b)
""",

"cc_618_B. Guess the Permutation": """\
import random
random.seed(42)
n = 50
perm = list(range(1, n+1))
random.shuffle(perm)
print(n)
for i in range(n):
    row = [0]*n
    for j in range(n):
        if i != j:
            row[j] = max(perm[k] for k in range(min(i,j), max(i,j)+1))
    print(*row)
""",

"cc_69_A. Young Physicist": """\
import random
random.seed(42)
n = 100
vecs = [[random.randint(-100,100) for _ in range(3)] for _ in range(n-1)]
last = [-sum(v[k] for v in vecs) for k in range(3)]
vecs.append(last)
random.shuffle(vecs)
print(n)
for v in vecs:
    print(*v)
""",

"cc_198_B. Jumping on Walls": """\
import random
random.seed(42)
n, k = 50000, 1
print(n, k)
print(''.join(random.choices('.-', k=n)))
print(''.join(random.choices('.-', k=n)))
""",

"cc_580_B. Kefa and Company": """\
import random
random.seed(42)
n, d = 50000, 1
print(n, d)
for _ in range(n):
    print(random.randint(1,10**9), random.randint(1,10**9))
""",

"cc_436_B. Om Nom and Spiders": """\
import random
random.seed(42)
n, m, k = 500, 500, 10000
print(n, m, k)
for _ in range(n):
    print(''.join(random.choices('UDL R.', weights=[1,1,1,1,6], k=m).replace(' ','')))
""",

"cc_436_B. Om Nom and Spiders": """\
import random
random.seed(42)
n, m, k = 500, 500, 10000
print(n, m, k)
for _ in range(n):
    row = ''.join(random.choices('UDLR.', weights=[1,1,1,1,6], k=m))
    print(row)
""",

"cc_1070_D. Garbage Disposal": """\
import random
random.seed(42)
n, k = 100000, 10
print(n, k)
print(*[random.randint(0, 10**9) for _ in range(n)])
""",

"cc_1016_A. Death Note": """\
import random
random.seed(42)
n, m = 100000, 10**9
print(n, m)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_483_C. Diverse Permutation": """\
print(100000, 99999)
""",

"cc_122_D. Lucky Transformation": """\
import random
random.seed(42)
n, d = 50000, 10
print(n, d)
print(''.join(random.choices('47', k=n)))
""",

"cc_1144_B. Parity Alternated Deletions": """\
import random
random.seed(42)
n = 2000
print(n)
print(*[random.randint(1,10**9) for _ in range(n)])
""",

"cc_1294_B. Collecting Packages": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    n = 1000
    print(n)
    pts = sorted(set((random.randint(0,10**9), random.randint(0,10**9)) for _ in range(n)))[:n]
    for x,y in pts:
        print(x, y)
""",

"cc_1493_A. Anti-knapsack": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = random.randint(2, 10**9)
    k = random.randint(1, min(n, 10**9))
    print(n, k)
""",

"cc_602_A. Two Bases": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    bx = random.randint(2, 40)
    n = 10
    print(n, bx)
    print(*[random.randint(0, bx-1) for _ in range(n)])
""",

"cc_719_B. Anatoly and Cockroaches": """\
import random
random.seed(42)
n = 50000
print(n)
print(''.join(random.choices('rgb', k=n)))
""",

"cc_673_A. Bear and Game": """\
import random
random.seed(42)
n = 90
minutes = sorted(random.sample(range(1, 500), n))
print(n)
print(*minutes)
""",

"cc_924_A. Mystical Mosaic": """\
import random
random.seed(42)
n, m = 50, 50
print(n, m)
for _ in range(n):
    print(''.join(random.choices('.#', weights=[9,1], k=m)))
""",

"cc_248_A. Cupboards": """\
import random
random.seed(42)
n = 10000
print(n)
for _ in range(n):
    print(random.randint(0,1), random.randint(0,1))
""",

"cc_1392_B. Omkar and Infinity Clock": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = 20
    b = random.randint(1, 10**9)
    print(n, b)
    print(*[random.randint(-10**9, 10**9) for _ in range(n)])
""",

"cc_1450_B. Balls of Steel": """\
import random
random.seed(42)
t = 1
n, k = 1000, 3
print(t)
print(n, k)
for _ in range(n):
    print(random.randint(-10**9, 10**9), random.randint(-10**9, 10**9))
""",

"cc_1408_B. Arrays Sum": """\
import random
random.seed(42)
t = 1
n, m = 50000, 25
print(t)
print(n, m)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_245_D. Restoring Table": """\
import random
random.seed(42)
n = 100
print(n)
for i in range(n):
    row = [random.randint(0, 10**9) if i!=j else -1 for j in range(n)]
    print(*row)
""",

"cc_1271_A. Suits": """\
print(100000)
print(100000)
print(100100)
print(100000)
print(100)
""",

"cc_157_A. Game Outcome": """\
import random
random.seed(42)
n = 12
print(n)
for _ in range(n):
    print(*[random.randint(1,100) for _ in range(n)])
""",

# ── NEW BATCH ────────────────────────────────────────────────────────────────

"cc_634_C. Factory Repairs": """\
import random
random.seed(42)
n = 100000
b = 50
a = 100
k = 10
c = 100000
print(n, b, a, k, c)
for _ in range(c):
    if random.random() < 0.5:
        d = random.randint(1, n)
        ai = random.randint(0, 500)
        print(1, d, ai)
    else:
        d = random.randint(1, n)
        print(2, d)
""",

"cc_643_B. Bear and Two Paths": """\
import random
random.seed(42)
n = 1000
m = n + 2
print(n, m)
a, b, c, d = random.sample(range(1, n+1), 4)
print(a, b, c, d)
""",

"cc_737_A. Road to Cinema": """\
import random
random.seed(42)
k = 100
t = 100
s = 10**6
v = 10**6
print(k, t, s, v)
positions = sorted(random.sample(range(1, s), k))
for pos in positions:
    print(pos, random.randint(1, 10**6))
print(*sorted([random.randint(1, 10**6) for _ in range(t)]))
""",

"cc_1420_D. Rescue Nibel!": """\
import random
random.seed(42)
n = 200000
m = 100000
print(n, m)
for _ in range(n):
    l = random.randint(1, 10**9)
    r = l + random.randint(1, 10**6)
    print(l, r)
""",

"cc_80_C. Heroes": """\
import random
random.seed(42)
heroes = ["Anka", "Chapay", "Cleo", "Troll", "Dracul", "Snowy", "Hexadecimal"]
pairs = []
for i in range(42):
    a = heroes[i % 7]
    b = heroes[(i + 3) % 7]
    if a != b:
        pairs.append((a, b))
pairs = pairs[:42]
print(len(pairs))
for a, b in pairs:
    print(f"{a} likes {b}")
print(998244353, 1000000007, 999999937)
""",

"cc_1106_F. Lunar New Year and a Recursive Sequence": """\
import random
random.seed(42)
k = 5
print(k)
print(*[random.randint(0, 998244352) for _ in range(k)])
n = random.randint(k+1, 10**6)
v = random.randint(1, 998244352)
print(n, v)
""",

"cc_p00865 Expected Allowance": """\
import random
random.seed(42)
for _ in range(10):
    n = random.randint(1, 13)
    m = random.randint(2, 20)
    k = random.randint(-10, 10)
    print(n, m, k)
print(0, 0, 0)
""",

"cc_p00184 Tsuruga Castle": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(5, 50)
    print(n)
    for _ in range(n):
        print(random.randint(0, 120))
print(0)
""",

"cc_p00822 Weather Forecast": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(1, 16)
    print(n)
    for _ in range(n):
        print(*[random.randint(0, 1) for _ in range(16)])
print(0)
""",

"cc_p01499 Rabbit Game Playing": """\
import random
random.seed(42)
n = 100
T = 20
print(n, T)
for _ in range(n):
    print(random.randint(-1000, 1000))
print(0)
""",

"cc_p01437 Infinity Maze": """\
import random
random.seed(42)
for _ in range(2):
    H = random.randint(3, 8)
    W = random.randint(3, 8)
    T = random.randint(1, 1000)
    print(H, W, T)
    grid = [['.' for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if random.random() < 0.15:
                grid[r][c] = '#'
    r0, c0 = random.randint(0, H-1), random.randint(0, W-1)
    grid[r0][c0] = 'S'
    for row in grid:
        print(''.join(row))
print(0, 0, 0)
""",

"cc_544_E. Remembering Strings": """\
import random
random.seed(42)
n, m = 20, 5
print(n, m)
for _ in range(n):
    print(''.join(random.choices('abcde', k=m)))
for _ in range(n):
    print(*[random.randint(1, 10**6) for _ in range(m)])
""",

"cc_937_D. Sleepy Game": """\
import random
random.seed(42)
n = 1000
m = 700
has_edge = set(random.sample(range(1, n+1), m))
edge_to = {v: random.randint(1, n) for v in has_edge}
print(n, m)
for v in range(1, n+1):
    if v in has_edge:
        print(1, edge_to[v])
    else:
        print(0)
s = random.randint(1, n)
print(s)
""",

"cc_1030_E. Vasya and Good Sequences": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.randint(1, 10**18) for _ in range(n)])
""",

"cc_1075_D. Intersecting Subtrees": "",  # interactive — skip
"cc_p01293 Whist": "",  # complex card game — skip

"cc_1304_E. 1-Trees and Queries": """\
import random
random.seed(42)
n = 1000
print(n)
for i in range(2, n+1):
    print(random.randint(1, i-1), i)
q = 10000
print(q)
for _ in range(q):
    v1 = random.randint(1, n)
    u1 = random.randint(1, n)
    v2 = random.randint(1, n)
    u2 = random.randint(1, n)
    k = random.randint(1, 20)
    print(v1, u1, v2, u2, k)
""",

"cc_442_C. Artem and Array ": """\
import random
random.seed(42)
n = 200000
print(n)
print(*random.sample(range(1, 10**9), n))
""",

"cc_319_B. Psychos in a Line": """\
import random
random.seed(42)
n = 100000
perm = list(range(1, n+1))
random.shuffle(perm)
print(n)
print(*perm)
""",

"cc_903_D. Almost Difference": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_702_E. Analysis of Pathes in Functional Graph": """\
import random
random.seed(42)
n = 100000
k = 50
print(n, k)
print(*[random.randint(0, n-1) for _ in range(n)])
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_p01488 TransferTrain": """\
import random
random.seed(42)
n = 2
ti = 10
A, B = 'Alpha', 'Beta'
print(n, ti)
print(A, B)
k1 = 3
s1 = [A, 'Gamma', B]
print(k1)
print(*s1)
print(*[random.randint(10, 500) for _ in range(k1-1)])
k2 = 3
s2 = [A, 'Delta', B]
print(k2)
print(*s2)
print(*[random.randint(10, 500) for _ in range(k2-1)])
""",

"cc_p00749 Off Balance": """\
import random
random.seed(42)
for _ in range(3):
    w = random.randint(5, 9)
    h = random.randint(3, 5)
    grid = [['.' for _ in range(w)] for _ in range(h)]
    # bottom row: piece 1
    for x in range(w):
        grid[h-1][x] = '1'
    # row above bottom: some piece 2 cells (all directly above piece 1, so ground[2] gets set)
    for x in range(w):
        if random.random() < 0.4:
            grid[h-2][x] = '2'
    if not any(c == '2' for c in grid[h-2]):
        grid[h-2][w//2] = '2'
    print(w, h)
    for row in grid:
        print(''.join(row))
print(0, 0)
""",

"cc_p00887 Awkward Lights": """\
import random
random.seed(42)
for _ in range(3):
    M = random.randint(2, 5)
    N = random.randint(2, 5)
    D = random.randint(1, 2)
    print(M, N, D)
    for _ in range(N):
        print(*[random.choice([0, 1]) for _ in range(M)])
print(0, 0, 0)
""",

"cc_512_B. Fox And Jumping": """\
import random
random.seed(42)
n = 50
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_83_C. Track": """\
import random
random.seed(42)
n, m, k = 12, 12, 3
print(n, m, k)
letters = 'abcdef'
grid = [[random.choice(letters) for _ in range(m)] for _ in range(n)]
si, sj = random.randint(0, n-1), random.randint(0, m-1)
ti, tj = random.randint(0, n-1), random.randint(0, m-1)
while (ti, tj) == (si, sj):
    ti, tj = random.randint(0, n-1), random.randint(0, m-1)
grid[si][sj] = 'S'
grid[ti][tj] = 'T'
for row in grid:
    print(''.join(row))
""",

"cc_1199_E. Matching vs Independent Set": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(5, 20)
    m = random.randint(n, 3*n)
    print(n, m)
    edges = set()
    while len(edges) < m:
        u = random.randint(1, 3*n)
        v = random.randint(1, 3*n)
        if u != v and (u,v) not in edges and (v,u) not in edges:
            edges.add((u, v))
    for u, v in edges:
        print(u, v)
""",

"cc_45_H. Road Problem": """\
import random
random.seed(42)
n = 50
m = 70
print(n, m)
edges = set()
# ensure connectivity first
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((j, i))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
for u, v in edges:
    print(u, v)
""",

"cc_675_E. Trains and Statistic": """\
import random
random.seed(42)
n = 100000
print(n)
# a[i] (1-indexed, for i in 1..n-1) must be in [i+1, n]
vals = [random.randint(i+1, n) for i in range(1, n)]
print(*vals)
""",

"cc_954_F. Runner's Problem": """\
import random
random.seed(42)
n = 50
m = 100000
print(n, m)
for _ in range(n):
    row = random.randint(1, 3)
    l = random.randint(1, m-1)
    r = random.randint(l, m)
    print(row, l, r)
""",

"cc_9_E. Interesting Graph and Apples": """\
import random
random.seed(42)
n = 100
m = 150
print(n, m)
edges = set()
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((j, i))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
for u, v in edges:
    print(u, v)
""",

"cc_1206_F. Almost All": """\
import random
random.seed(42)
n = 100
print(n)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

"cc_1332_G. No Monotone Triples": """\
import random
random.seed(42)
n = 30
q = 28
print(n, q)
print(*[random.randint(1, 10**9) for _ in range(n)])
for _ in range(q):
    l = random.randint(1, n-1)
    r = random.randint(l+1, n)
    print(l, r)
""",

"cc_799_D. Field expansion": """\
import random
random.seed(42)
h = random.randint(2, 100)
w = random.randint(2, 100)
c = random.randint(h+1, 10**9)
b = random.randint(w+1, 10**9)
n = random.randint(1, 34)
d = [random.randint(2, 10**6) for _ in range(n)]
print(c, b, h, w, n)
print(*d)
""",

"cc_1131_E. String Multiplication": """\
import random, string
random.seed(42)
n = 5
print(n)
for _ in range(n):
    length = random.randint(2, 20)
    print(''.join(random.choices(string.ascii_lowercase, k=length)))
""",

"cc_1251_E1. Voting (Easy Version)": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(5, 20)
    print(n)
    for _ in range(n):
        mi = random.randint(0, n)
        pi = random.randint(1, 100)
        print(mi, pi)
""",

"cc_521_C. Pluses everywhere": """\
import random
random.seed(42)
n = 100
k = 50
print(n, k)
# n digits printed without spaces
print(''.join(str(random.randint(0,9)) for _ in range(n)))
""",

"cc_638_D. Three-dimensional Turtle Super Computer ": """\
import random
random.seed(42)
n, m, k = 4, 5, 3
print(n, m, k)
for x in range(n):
    for y in range(m):
        print(''.join(str(random.randint(0,1)) for _ in range(k)))
    if x < n-1:
        print()
""",

"cc_733_C. Epidemic in Monstropolis": """\
import random
random.seed(42)
n = 500
a = [random.randint(1, 10**9) for _ in range(n)]
m = random.randint(1, n)
b = [random.randint(1, 10**9) for _ in range(m)]
print(n)
print(*a)
print(m)
print(*b)
""",

"cc_825_E. Minimal Labels": """\
import random
random.seed(42)
n = 100
m = 150
print(n, m)
# DAG: always edge from higher node to lower node
edges = set()
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((i, j))
while len(edges) < m:
    u = random.randint(2, n)
    v = random.randint(1, u-1)
    if (u, v) not in edges:
        edges.add((u, v))
for u, v in edges:
    print(u, v)
""",

"cc_993_D. Compute Power": """\
import random
random.seed(42)
n = 50
print(n)
print(*[random.randint(1, 100) for _ in range(n)])
print(*[random.randint(1, 100) for _ in range(n)])
""",

"cc_1144_F. Graph Without Long Directed Paths": """\
import random
random.seed(42)
n = 100
m = 150
print(n, m)
edges = set()
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((j, i))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
for u, v in edges:
    print(u, v)
""",

"cc_1508_C. Complete the MST": """\
import random
random.seed(42)
n = 6
m = 12
print(n, m)
edges = set()
# ensure connectivity
for i in range(2, n+1):
    j = random.randint(1, i-1)
    w = random.randint(1, 10**7)
    edges.add((j, i, w))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and not any(e[0]==min(u,v) and e[1]==max(u,v) for e in edges):
        w = random.randint(1, 10**7)
        edges.add((min(u,v), max(u,v), w))
for u, v, w in edges:
    print(u, v, w)
""",

"cc_840_B. Leha and another game about graph": """\
import random
random.seed(42)
n = 50
m = 70
print(n, m)
print(*[-1]*n)
edges = set()
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((j, i))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
for u, v in edges:
    print(u, v)
""",

"cc_1000_C. Covered Points Count": """\
import random
random.seed(42)
n = 100
print(n)
for _ in range(n):
    l = random.randint(1, 10**15)
    r = random.randint(l, l + 10**9)
    print(l, r)
""",

"cc_1025_D. Recovering BST": """\
import random
random.seed(42)
n = 30
vals = sorted(random.sample(range(1, 10**6), n))
print(n)
print(*vals)
""",

"cc_1139_E. Maximize Mex": """\
import random
random.seed(42)
n, m = 20, 20
print(n, m)
p = list(range(n)); random.shuffle(p)
print(*p)
c = list(range(1, m+1)); random.shuffle(c)
print(*c)
d = random.randint(1, n//2)
print(d)
disabled = random.sample(range(1, n+1), d)
for x in disabled:
    print(x)
""",

"cc_1157_E. Minimum Array": """\
import random
random.seed(42)
n = 50
print(n)
print(*[random.randint(0, 10**9) for _ in range(n)])
print(*[random.randint(0, n-1) for _ in range(n)])
""",

"cc_1198_C. Matching vs Independent Set": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    N = random.randint(5, 20)
    M = random.randint(N, 3*N)
    print(N, M)
    edges = set()
    while len(edges) < M:
        x = random.randint(1, 3*N)
        y = random.randint(1, 3*N)
        if x != y and (x,y) not in edges and (y,x) not in edges:
            edges.add((x, y))
    for x, y in edges:
        print(x, y)
""",

"cc_1256_E. Yet Another Division Into Teams": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_1406_C. Link Cut Centroids": """\
import random
random.seed(42)
T = 5
print(T)
for _ in range(T):
    n = random.randint(3, 15)
    print(n)
    for i in range(2, n+1):
        p = random.randint(1, i-1)
        print(p, i)
""",

"cc_1450_D. Rating Compression": """\
import random
random.seed(42)
T = 5
print(T)
for _ in range(T):
    n = random.randint(3, 20)
    print(n)
    print(*[random.randint(1, n) for _ in range(n)])
""",

"cc_1523_D. Love-Hate": """\
import random
random.seed(42)
n, m, p = 20, 15, 15
print(n, m, p)
for _ in range(n):
    print(''.join(str(random.randint(0,1)) for _ in range(m)))
""",

"cc_270_D. Greenhouse Effect": """\
import random
random.seed(42)
n, m = 20, 15
print(n, m)
for _ in range(n):
    c = random.randint(1, m)
    x = round(random.uniform(1.0, 10**5), 6)
    print(c, x)
""",

"cc_459_E. Pashmak and Graph": """\
import random
random.seed(42)
n, m = 100, 120
print(n, m)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    while u == v:
        v = random.randint(1, n)
    w = random.randint(1, 100000)
    print(u, v, w)
""",

"cc_505_D. Mr. Kitayuta's Technology": """\
import random
random.seed(42)
n, m = 50, 60
print(n, m)
for _ in range(m):
    a = random.randint(1, n)
    b = random.randint(1, n)
    print(a, b)
""",

"cc_554_E. Love Triangles": """\
import random
random.seed(42)
n = 100
k = 50
print(n, k)
edges = set()
while len(edges) < k:
    a = random.randint(1, n)
    b = random.randint(1, n)
    if a != b and (a,b) not in edges and (b,a) not in edges:
        edges.add((a, b))
for a, b in edges:
    print(a, b, 1)
""",

"cc_580_D. Kefa and Dishes": """\
import random
random.seed(42)
n, m, k = 10, 8, 5
print(n, m, k)
print(*[round(random.uniform(1.0, 10**9), 2) for _ in range(n)])
used = set()
while len(used) < k:
    x = random.randint(1, n)
    y = random.randint(1, m)
    if (x, y) not in used:
        used.add((x, y))
        z = random.randint(1, 10**8)
        print(x, y, z)
""",

"cc_602_C. The Two Routes": """\
import random
random.seed(42)
n = 20
m = 15
print(n, m)
edges = set()
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((min(i,j), max(i,j)))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v:
        edges.add((min(u,v), max(u,v)))
for a, b in list(edges)[:m]:
    print(a, b)
""",

"cc_808_G. Anthem of Berland": """\
import random, string
random.seed(42)
t_len = 10
s_len = 50
t = ''.join(random.choices(string.ascii_lowercase[:5], k=t_len))
s = list(''.join(random.choices(string.ascii_lowercase[:5], k=s_len)))
# replace some positions with '?'
for i in range(s_len):
    if random.random() < 0.3:
        s[i] = '?'
print(''.join(s))
print(t)
""",

"cc_924_C. Riverside Curio": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(0, 20) for _ in range(n)])
""",

"cc_1323_D. Present": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(0, 10**9) for _ in range(n)])
""",

"cc_177_C1. Party": """\
import random
random.seed(42)
n = 15
print(n)
all_pairs = [(i,j) for i in range(1,n+1) for j in range(i+1,n+1)]
random.shuffle(all_pairs)
k1 = random.randint(5, min(20, len(all_pairs)))
friends = all_pairs[:k1]
print(k1)
for u,v in friends:
    print(u, v)
k2 = random.randint(3, min(10, len(all_pairs)-k1))
enemies = all_pairs[k1:k1+k2]
print(k2)
for u,v in enemies:
    print(u, v)
""",

"cc_270_B. Multithreading": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_341_B. Bubble Sort Graph": """\
import random
random.seed(42)
n = 100
perm = list(range(1, n+1))
random.shuffle(perm)
print(n)
print(*perm)
""",

"cc_505_B. Mr. Kitayuta's Colorful Graph": """\
import random
random.seed(42)
n, m = 20, 30
print(n, m)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    c = random.randint(1, 5)
    print(u, v, c)
q = 10
print(q)
for _ in range(q):
    u = random.randint(1, n)
    v = random.randint(1, n)
    print(u, v)
""",

"cc_698_B. Fix a Tree": """\
import random
random.seed(42)
n = 50
parents = [random.randint(1, n) for _ in range(n)]
# force one root
parents[0] = 1
print(n)
print(*parents)
""",

"cc_808_E. Selling Souvenirs": """\
import random
random.seed(42)
n = 50
m = 1000
print(n, m)
for _ in range(n):
    w = random.randint(1, 3)
    c = random.randint(1, 1000)
    print(w, c)
""",

"cc_878_B. Teams Formation": """\
import random
random.seed(42)
n = 100
k = 5
m = 10**9
print(n, k, m)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_901_D. Weighting a Tree": """\
import random
random.seed(42)
n = 40
m = 50
print(n, m)
print(*[random.randint(-10**9, 10**9) for _ in range(n)])
edges = set()
for i in range(2, n+1):
    j = random.randint(1, i-1)
    edges.add((j, i))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
for u, v in list(edges)[:m]:
    print(u, v)
""",

"cc_979_E. Kuro and Topological Parity": """\
import random
random.seed(42)
n = 50
p = 1
print(n, p)
print(*[random.choice([-1, 0, 1]) for _ in range(n)])
""",

"cc_999_F. Cards and Joy": """\
import random
random.seed(42)
n = 50
k = 3
print(n, k)
# c: card values each player holds
print(*[random.randint(1, 20) for _ in range(n)])
# f: favorite card value for each player
print(*[random.randint(1, 20) for _ in range(n)])
# h: joy values for 1..k matching cards
print(*sorted([random.randint(1, 10**6) for _ in range(k)]))
""",

"cc_1140_C. Playlist": """\
import random
random.seed(42)
n = 100
k = 10
print(n, k)
for _ in range(n):
    t = random.randint(1, 10**5)
    b = random.randint(1, 10**9)
    print(t, b)
""",

"cc_1325_C. Ehab and Path-etic MEXs": """\
import random
random.seed(42)
n = 100
print(n)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

"cc_1525_E. Assimilation IV": """\
import random
random.seed(42)
n, m = 5, 8
print(n, m)
for _ in range(n):
    print(*[random.randint(1, 10**9) for _ in range(m)])
""",

"cc_272_C. Dima and Staircase": """\
import random
random.seed(42)
m = 10
print(m)
widths = sorted([random.randint(1, 100) for _ in range(m)])
print(*widths)
t = 20
print(t)
for _ in range(t):
    w = random.randint(1, m)
    h = random.randint(1, 10**9)
    print(w, h)
""",

"cc_319_C. Kalila and Dimna in the Logging Industry": """\
import random
random.seed(42)
n = 50
a = sorted(random.sample(range(1, 10**6), n))
b = sorted([random.randint(1, 10**6) for _ in range(n-1)], reverse=True) + [0]
print(n)
print(*a)
print(*b)
""",

"cc_556_D. Case of Fugitive": """\
import random
random.seed(42)
n = 20
m = 10
print(n, m)
for _ in range(n):
    l = random.randint(1, 100)
    r = random.randint(l, l + 50)
    print(l, r)
print(*[random.randint(1, 150) for _ in range(m)])
""",

"cc_582_B. Once Again...": """\
import random
random.seed(42)
n = 99
m = 10000000
print(n, m)
print(*[random.randint(0, 300) for _ in range(n)])
""",

"cc_1315_B. Homecoming": """\
import random
random.seed(42)
T = 5
print(T)
for _ in range(T):
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    p = random.randint(10, 50)
    length = random.randint(5, 20)
    print(a, b, p)
    print(''.join(random.choices('AB', k=length)))
""",

"cc_1359_B. New Theatre Square": """\
import random
random.seed(42)
T = 4
print(T)
for _ in range(T):
    n = random.randint(1, 5)
    m = random.randint(1, 10)
    x = random.randint(1, 100)
    y = random.randint(1, 200)
    print(n, m, x, y)
    for _ in range(n):
        print(''.join(random.choices('.*', weights=[3,1], k=m)))
""",

"cc_1379_D. New Passenger Trams": """\
import random
random.seed(42)
n = 10
h = 24
m = 60
k = 8
print(n, h, m, k)
for _ in range(n):
    H = random.randint(0, h-1)
    M = random.randint(0, m-1)
    print(H, M)
""",

"cc_1515_G. Phoenix and Odometers": """\
import random
random.seed(42)
n, m = 10, 15
print(n, m)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    w = random.randint(0, 10)
    print(u, v, w)
q = 5
print(q)
for _ in range(q):
    u = random.randint(1, n)
    v = random.randint(1, n)
    w = random.randint(1, 20)
    print(u, v, w)
""",

"cc_238_D. Tape Programming": """\
import random, string
random.seed(42)
n = 30
q = 20
print(n, q)
chars = string.digits + '<>'
print(''.join(random.choices(chars, k=n)))
for _ in range(q):
    l = random.randint(1, n-1)
    r = random.randint(l, n)
    print(l, r)
""",

"cc_617_E. XOR and Favorite Number": """\
import random
random.seed(42)
n, m = 100, 50
k = random.randint(1, 1000)
print(n, m, k)
# values must be < 2^20 so prefix XOR stays within count array bounds
print(*[random.randint(0, (1<<20)-1) for _ in range(n)])
for _ in range(m):
    l = random.randint(1, n-1)
    r = random.randint(l, n)
    print(l, r)
""",

"cc_638_B. Making Genome in Berland": """\
import random
random.seed(42)
# generate parts from a cycle so Eulerian path exists
n = 10
alpha = 'abcd'
k = len(alpha)
# build an Eulerian cycle through alpha
cycle = [random.randint(0,k-1) for _ in range(n)]
cycle.append(cycle[0])  # close the cycle
parts = [alpha[cycle[i]] + alpha[cycle[i+1]] for i in range(n)]
print(n)
for p in parts:
    print(p)
""",

"cc_1324_D. Pair of Topics": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_1365_D. Solve The Maze": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n, m = random.randint(3, 6), random.randint(3, 6)
    print(n, m)
    grid = [['.' for _ in range(m)] for _ in range(n)]
    # place one G at bottom-right
    grid[n-1][m-1] = 'G'
    # place a few B and # at random
    for _ in range(2):
        r, c = random.randint(0, n-1), random.randint(0, m-2)
        if grid[r][c] == '.':
            grid[r][c] = 'B'
    for _ in range(2):
        r, c = random.randint(0, n-2), random.randint(0, m-1)
        if grid[r][c] == '.':
            grid[r][c] = '#'
    for row in grid:
        print(''.join(row))
""",

"cc_156_B. Suspects": """\
import random
random.seed(42)
n = 20
m = 5
print(n, m)
for _ in range(n):
    sign = random.choice(['+', '-'])
    val = random.randint(1, n)
    print(f'{sign}{val}')
""",

"cc_741_A. Arpa's loud Owf and Mehrdad's evil plan": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(1, n) for _ in range(n)])
""",

"cc_1027_D. Mouse Hunt": """\
import random
random.seed(42)
n = 50
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
print(*[random.randint(1, n) for _ in range(n)])
""",

"cc_1199_D. Welfare State": """\
import random
random.seed(42)
n = 50
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
q = 20
print(q)
for _ in range(q):
    t = random.randint(1, 2)
    if t == 1:
        p = random.randint(1, n)
        x = random.randint(1, 10**9)
        print(t, p, x)
    else:
        x = random.randint(1, 10**9)
        print(t, x)
""",

"cc_1343_F. Restore the Permutation by Sorted Segments": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(3, 7)
    print(n)
    perm = list(range(1, n+1))
    random.shuffle(perm)
    # each segment is a single element (singleton) — trivially valid
    for i in range(1, n):
        print(1, perm[i])
""",

"cc_1385_G. Columns Swaps": """\
import random
random.seed(42)
T = 4
print(T)
for _ in range(T):
    n = random.randint(3, 8)
    print(n)
    row1 = list(range(1, n+1)); random.shuffle(row1)
    row2 = list(range(1, n+1)); random.shuffle(row2)
    print(*row1)
    print(*row2)
""",

"cc_1475_C. Ball in Berland": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    a = random.randint(3, 10)
    b = random.randint(3, 10)
    k = random.randint(2, 8)
    print(a, b, k)
    print(*[random.randint(1, a) for _ in range(k)])
    print(*[random.randint(1, b) for _ in range(k)])
""",

"cc_1525_C. Robot Collisions": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(5, 20)
    m = random.randint(n*2, n*5)
    print(n, m)
    positions = sorted(random.sample(range(1, m+1), n))
    print(*positions)
    print(*[random.choice(['L', 'R']) for _ in range(n)])
""",

"cc_223_B. Two Strings": """\
import random, string
random.seed(42)
n = 50
s = ''.join(random.choices(string.ascii_lowercase, k=n))
t = ''.join(random.choices(string.ascii_lowercase, k=n))
print(s)
print(t)
""",

"cc_366_C. Dima and Salad": """\
import random
random.seed(42)
n, k = 50, 3
print(n, k)
# values must be small so d=u-k*v stays within [0,100000]
print(*[random.randint(1, 100) for _ in range(n)])
print(*[random.randint(1, 30) for _ in range(n)])
""",

"cc_675_D. Tree Construction": """\
import random
random.seed(42)
n = 100
vals = sorted(random.sample(range(1, 10**9), n))
print(n)
print(*vals)
""",

"cc_980_D. Perfect Groups": """\
import random
random.seed(42)
n = 80
print(n)
print(*[random.randint(-10**4, 10**4) for _ in range(n)])
""",

"cc_1096_F. Inversion Expectation": """\
import random
random.seed(42)
n = 50
known_positions = random.sample(range(1, n+1), n//2)
vals = [-1] * n
perm_vals = random.sample(range(1, n+1), n//2)
for pos, val in zip(known_positions, perm_vals):
    vals[pos-1] = val
print(n)
print(*vals)
""",

"cc_1244_D. Paint the Tree": """\
import random
random.seed(42)
n = 15
print(n)
for _ in range(3):
    print(*[random.randint(1, 100) for _ in range(n)])
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

# ── batch 6 ──────────────────────────────────────────────────────────────────

"cc_1393_C. Pinkie Pie Eats Patty-cakes": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    n = random.randint(6, 12)
    # small range ensures duplicates so maxi >= 2 (avoids division-by-zero)
    vals = [random.randint(1, 4) for _ in range(n)]
    print(n)
    print(*vals)
""",

"cc_34_E. Collisions": """\
import random
random.seed(42)
n, maxTime = 8, 100
print(n, maxTime)
positions = sorted(random.sample(range(-100, 100), n))
for x in positions:
    v = random.randint(-10, 10)
    m = random.randint(1, 20)
    print(x, v, m)
""",

"cc_467_C. George and Job": """\
import random
random.seed(42)
n, m, k = 50, 5, 8
print(n, m, k)
print(*[random.randint(0, 100) for _ in range(n)])
""",

"cc_513_G1. Inversions problem": """\
import random
random.seed(42)
n, k = 5, 2
p = list(range(1, n+1))
random.shuffle(p)
print(n, k)
print(*p)
""",

"cc_682_D. Alyona and Strings": """\
import random
random.seed(42)
n, m, k = 20, 20, 3
s1 = ''.join(random.choices('abcd', k=n))
s2 = ''.join(random.choices('abcd', k=m))
print(n, m, k)
print(s1)
print(s2)
""",

"cc_705_C. Thor": """\
import random
random.seed(42)
n_apps = 5
q = 30
print(n_apps, q)
for _ in range(q):
    op = random.randint(1, 3)
    x = random.randint(1, n_apps) if op <= 2 else 0
    print(op, x)
""",

"cc_911_D. Inversion Counting": """\
import random
random.seed(42)
n = 12
p = list(range(1, n+1))
random.shuffle(p)
print(n)
print(*p)
m = 10
print(m)
for _ in range(m):
    l = random.randint(1, n-1)
    r = random.randint(l, n)
    print(l, r)
""",

"cc_1360_G. A/B Matrix": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    n = random.randint(2, 5)
    m = random.randint(2, 5)
    a = random.randint(1, m)
    b = random.randint(1, n)
    print(n, m, a, b)
""",

"cc_1445_D. Divide and Sum": """\
import random
random.seed(42)
n = 10
print(n)
print(*[random.randint(1, 10**9) for _ in range(2*n)])
""",

"cc_476_E. Dreamoon and Strings": """\
import random
random.seed(42)
n, m = 30, 5
s = ''.join(random.choices('abc', k=n))
p = ''.join(random.choices('abc', k=m))
print(s)
print(p)
""",

"cc_620_D. Professor GukiZ and Two Arrays": """\
import random
random.seed(42)
n = 10
a = [random.randint(-1000, 1000) for _ in range(n)]
m = 10
b = [random.randint(-1000, 1000) for _ in range(m)]
print(n)
print(*a)
print(m)
print(*b)
""",

"cc_641_C. Little Artem and Dance": """\
import random
random.seed(42)
n, q = 20, 30
print(n, q)
for _ in range(q):
    if random.random() < 0.6:
        x = random.randint(-n, n)
        print(1, x)
    else:
        print(2)
""",

"cc_780_D. Innokenty and a Football League": """\
n = 10
first_names = ['Alexander','Benjamin','Charles','Daniel','Edward',
               'Franklin','George','Henry','Isaac','James']
last_names  = ['Brown','Clark','Davis','Evans','Fisher',
               'Garcia','Harris','Irving','Johnson','Kumar']
print(n)
for i in range(n):
    print(first_names[i], last_names[i])
""",

"cc_1413_D. Shurikens": """\
n = 8
print(n)
# n pushes then n pops in increasing order (valid sequence)
for i in range(n):
    print('+')
for i in range(1, n+1):
    print('-', i)
""",

"cc_29_C. Mail Stamps": """\
n = 8
print(n)
# Chain: 1-2-3-...-9
for i in range(1, n+1):
    print(i, i+1)
""",

"cc_347_D. Lucky Common Subsequence": """\
import random
random.seed(42)
n, m = 30, 30
s1 = ''.join(random.choices('ABCDEFGHIJ', k=n))
s2 = ''.join(random.choices('ABCDEFGHIJ', k=m))
virus = ''.join(random.choices('ABCDEFGHIJ', k=4))
print(s1)
print(s2)
print(virus)
""",

"cc_442_A. Borya and Hanabi": """\
import random
random.seed(42)
n = 50
colors = 'RGBYW'
digits = '12345'
cards = [random.choice(colors) + random.choice(digits) for _ in range(n)]
print(n)
print(' '.join(cards))
""",

"cc_464_B. Restore Cube ": """\
# 8 vertices of a cube, each line sorted
s = 1000000
corners = [
    (-s,-s,-s),(-s,-s, s),(-s, s,-s),(-s, s, s),
    ( s,-s,-s),( s,-s, s),( s, s,-s),( s, s, s),
]
for c in corners:
    print(*sorted(c))
""",

"cc_536_A. Tavas and Karafs": """\
import random
random.seed(42)
A, B, n = 10, 5, 8
print(A, B, n)
for _ in range(n):
    l = random.randint(1, 10)
    t = random.randint(A, A + 50*B)
    m = random.randint(1, 20)
    print(l, t, m)
""",

"cc_586_D. Phillip and Trains": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n, k = 20, 5
    print(n, k)
    for _ in range(3):
        row = ''.join(random.choice('ABCDE.') for _ in range(n))
        print(row)
""",

# ── batch 5 ──────────────────────────────────────────────────────────────────

"cc_177_F1. Script Generation": """\
import random
random.seed(42)
n, k, T = 5, 15, 8
print(n, k, T)
segs = list(range(1, n+1))
for _ in range(k):
    f = random.randint(1, n)
    s = random.randint(1, n)
    v = random.randint(1, 50)
    print(f, s, v)
""",

"cc_222_D. Olympiad": """\
import random
random.seed(42)
n, x = 10, 1000
print(n, x)
print(*[random.randint(1, 500) for _ in range(n)])
print(*[random.randint(1, 500) for _ in range(n)])
""",

"cc_409_G. On a plane": """\
import random
random.seed(42)
n = 20
print(n)
for _ in range(n):
    x = round(random.uniform(-100.0, 100.0), 2)
    y = round(random.uniform(-100.0, 100.0), 2)
    print(x, y)
""",

"cc_71_D. Solitaire": """\
import random
random.seed(42)
ranks = '23456789TJQKA'
suits = 'CDHS'
n, m = 5, 10  # 50 cells, using 50 of 52 regular cards
# Two valid 3x3 same-suit blocks
block1 = ['2H','3H','4H','5H','6H','7H','8H','9H','TH']  # rows 0-2, cols 0-2
block2 = ['2D','3D','4D','5D','6D','7D','8D','9D','TD']  # rows 0-2, cols 6-8
all_cards = [r+s for r in ranks for s in suits]
used = set(block1 + block2)
remaining = [c for c in all_cards if c not in used]
random.shuffle(remaining)
grid = [[None]*m for _ in range(n)]
for i in range(3):
    for j in range(3):
        grid[i][j] = block1[i*3+j]
        grid[i][j+6] = block2[i*3+j]
rest = [(r,c) for r in range(n) for c in range(m) if grid[r][c] is None]
for idx, (r,c) in enumerate(rest):
    grid[r][c] = remaining[idx]
print(n, m)
for row in grid:
    print(' '.join(row))
""",

"cc_925_A. Stairs and Elevators": """\
import random
random.seed(42)
n, m = 20, 15  # n floors, m rooms per floor
n_stairs, n_elevators, v = 3, 5, 3
print(n, m, n_stairs, n_elevators, v)
stair_floors = sorted(random.sample(range(1, n+1), n_stairs))
print(*stair_floors)
elevator_floors = sorted(random.sample([f for f in range(1, n+1) if f not in stair_floors], n_elevators))
print(*elevator_floors)
queries = 20
print(queries)
for _ in range(queries):
    x1 = random.randint(1, m)
    x2 = random.randint(1, m)
    y1 = random.randint(1, n)
    y2 = random.randint(1, n)
    print(x1, y1, x2, y2)
""",

"cc_1216_C. White Sheet": """\
import random
random.seed(42)
# 3 rectangles; first is "white sheet", other two may cover it
def rect():
    x1 = random.randint(0, 500000)
    y1 = random.randint(0, 500000)
    x2 = random.randint(x1+1, 1000000)
    y2 = random.randint(y1+1, 1000000)
    return x1, y1, x2, y2
print(*rect())
print(*rect())
print(*rect())
""",

"cc_1500_A. Going Home": """\
import random
random.seed(42)
n = 60
# Include 4 equal values to ensure YES answer
vals = [random.randint(1, 2500000) for _ in range(n-4)]
target = random.randint(1, 2500000)
vals += [target, target, target, target]
random.shuffle(vals)
print(n)
print(*vals)
""",

"cc_954_E. Water Taps": """\
import random
random.seed(42)
n, t = 20, 50
print(n, t)
xs = sorted(random.sample(range(1, 200), n))
print(*xs)
print(*[random.randint(1, 100) for _ in range(n)])
""",

"cc_1184_C1. Heidi and the Turing Test (Easy)": """\
import random
random.seed(42)
n = 8  # reads 4n+1 = 33 points
x1, x2, y1, y2 = 5, 40, 5, 40
pts = []
# n points on each side (no corners shared)
ys_left = sorted(random.sample(range(y1+1, y2), n))
for y in ys_left:
    pts.append((x1, y))
ys_right = sorted(random.sample(range(y1+1, y2), n))
for y in ys_right:
    pts.append((x2, y))
# bottom: n points including corners
xs_bot = sorted(random.sample(range(x1, x2+1), n))
for x in xs_bot:
    pts.append((x, y1))
# top: n points including corners
xs_top = sorted(random.sample(range(x1, x2+1), n))
for x in xs_top:
    pts.append((x, y2))
# one wrong point not on border
wrong = (20, 20)
pts.append(wrong)
random.shuffle(pts)
print(n)
for x, y in pts:
    print(x, y)
""",

"cc_656_F. Ace It!": """\
import random
random.seed(42)
digits = [str(random.randint(2, 9)) for _ in range(20)]
print('A' + ''.join(digits))
""",

"cc_105_B. Dark Assembly": """\
import random
random.seed(42)
n, k, A = 10, 3, 5000
print(n, k, A)
for _ in range(n):
    lvl = random.randint(1, 9999)
    loy = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(lvl, loy)
""",

"cc_1129_A1. Toy Train (Simplified)": """\
import random
random.seed(42)
n, m = 50, 30
print(n, m)
for _ in range(m):
    a = random.randint(1, n)
    b = random.randint(1, n)
    print(a, b)
""",

"cc_867_C. Ordering Pizza": """\
import random
random.seed(42)
n, spp = 20, 8
print(n, spp)
for _ in range(n):
    t = random.randint(1, 10)   # slices wanted
    p1 = random.randint(1, 100) # price option 1
    p2 = random.randint(1, 100) # price option 2
    print(t, p1, p2)
""",

"cc_962_E. Byteland, Berland and Disputed Cities": """\
import random
random.seed(42)
n = 20
print(n)
xs = sorted(random.sample(range(-10000, 10000), n))
for x in xs:
    code = random.choice(['R', 'P', 'B'])
    print(x, code)
""",

"cc_1137_A. Skyscrapers": """\
import random
random.seed(42)
n, m = 5, 8
print(n, m)
for _ in range(n):
    print(*[random.randint(1, 1000) for _ in range(m)])
""",

"cc_1254_B1. Send Boxes to Alice (Easy Version)": """\
import random
random.seed(42)
n = 100
# Ensure at least 2 ones (so answer is not -1)
vals = [0]*n
ones_pos = random.sample(range(n), 30)
for i in ones_pos:
    vals[i] = 1
print(n)
print(*vals)
""",

"cc_32_D. Constellation": """\
import random
random.seed(42)
n, m, k = 15, 15, 3
grid = [['.']*m for _ in range(n)]
# Valid constellation at center (7,7) radius 3
cx, cy, rad = 7, 7, 3
for dr, dc in [(0,0),(rad,0),(-rad,0),(0,rad),(0,-rad)]:
    grid[cx+dr][cy+dc] = '*'
# Another constellation at (7, 11) radius 2 (non-overlapping)
cx2, cy2, rad2 = 7, 11, 2
for dr, dc in [(0,0),(rad2,0),(-rad2,0),(0,rad2),(0,-rad2)]:
    grid[cx2+dr][cy2+dc] = '*'
# Third constellation at (3, 3) radius 2
cx3, cy3, rad3 = 3, 3, 2
for dr, dc in [(0,0),(rad3,0),(-rad3,0),(0,rad3),(0,-rad3)]:
    grid[cx3+dr][cy3+dc] = '*'
print(n, m, k)
for row in grid:
    print(''.join(row))
""",

"cc_773_B. Dynamic Problem Scoring": """\
import random
random.seed(42)
n = 10  # n contestants (row 0=Vesya, row 1=Petya, rows 2..n-1=others)
print(n)
for i in range(n):
    vals = []
    for _ in range(5):
        if random.random() < 0.3:
            vals.append(-1)
        else:
            vals.append(random.randint(1, 100))
    print(*vals)
""",

"cc_818_C. Sofa Thief": """\
import random
random.seed(42)
n = 10
p, q = 5, 8  # read by ref but algorithm doesn't use them
print(n)
print(p, q)
sofas = []
for _ in range(n):
    x1 = random.randint(1, 20)
    y1 = random.randint(1, 20)
    x2 = random.randint(1, 20)
    y2 = random.randint(1, 20)
    sofas.append((x1, y1, x2, y2))
    print(x1, y1, x2, y2)
# co = feature vector of sofa 0
s = sofas[0]
lx, rx = min(s[0],s[2]), max(s[0],s[2])
ly, ry = min(s[1],s[3]), max(s[1],s[3])
c0 = sum(1 for a,b,c,d in sofas[1:] if min(a,c)>rx)
c1 = sum(1 for a,b,c,d in sofas[1:] if min(a,c)<lx)  # sofas with max_x < lx
c2 = sum(1 for a,b,c,d in sofas[1:] if min(b,d)>ry)
c3 = sum(1 for a,b,c,d in sofas[1:] if min(b,d)<ly)
print(c0, c1, c2, c3)
""",

"cc_1185_E. Polycarp and Snakes": """\
import random
random.seed(42)
t = 3
print(t)
for _ in range(t):
    n, m = random.randint(3, 6), random.randint(3, 6)
    print(n, m)
    # grid of dots and letters; snakes are contiguous regions
    grid = [['.']*m for _ in range(n)]
    # place a simple snake 'a' as a horizontal strip in row 0
    for c in range(m):
        grid[0][c] = 'a'
    for row in grid:
        print(''.join(row))
""",

# ── batch 4 ──────────────────────────────────────────────────────────────────

"cc_245_F. Log Stream Analysis": """\
# n m; then log lines until EOF. Output when m logs fall in n-1 second window.
import random
random.seed(42)
n, m = 30, 4
print(n, m)
# Generate m logs within 5 seconds at the end (so window triggers)
base = '2024-01-01 12:00:'
for i in range(8):
    sec = random.randint(0, 40)
    print(f'2024-01-01 12:{i:02d}:{sec:02d}: Log message {i}')
# Guarantee m logs within n-1=29 seconds
for i in range(m):
    print(f'2024-01-01 12:10:{i:02d}: Clustered message {i}')
""",

"cc_952_E. Cheese Board": """\
import random, string
random.seed(42)
n = 20
print(n)
for _ in range(n):
    name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3,8)))
    t = random.choice(['soft', 'hard'])
    print(name, t)
""",

"cc_1157_C2. Increasing Subsequence (hard version)": """\
import random
random.seed(42)
n = 30
print(n)
print(*[random.randint(1, 10) for _ in range(n)])
""",

"cc_1256_C. Platforms Jumping": """\
import random
random.seed(42)
n, m, d = 20, 8, 3
print(n, m, d)
# Platform widths: total of widths must equal n
# Each platform width c[i] in 1..6
c = []
total = 0
for i in range(m-1):
    w = random.randint(1, min(6, n - total - (m - i)))
    c.append(w)
    total += w
c.append(n - total)  # last platform fills the rest
print(*c)
""",

"cc_1342_C. Yet Another Counting Problem": """\
import random
random.seed(42)
t = 3
print(t)
for _ in range(t):
    a = random.randint(2, 10)
    b = random.randint(2, 10)
    q = random.randint(3, 8)
    print(a, b, q)
    for _ in range(q):
        l = random.randint(1, 50)
        r = random.randint(l, 200)
        print(l, r)
""",

"cc_1364_C. Ehab and Prefix MEXs": """\
import random
random.seed(42)
n = 8
print(n)
# a[i] must be non-decreasing and a[i] <= i+1 (1-indexed: a[i] in [0, i])
a = []
prev = 0
for i in range(n):
    # a[i] in [prev, i+1]  (allowing equal or larger)
    v = random.randint(prev, i + 1)
    a.append(v)
    prev = v
print(*a)
""",

"cc_149_B. Martian Clock": """\
# Format: "XXXXX:XXXXX" in base-36 (0-9A-Z)
import random
random.seed(42)
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def make_base36(length):
    return ''.join(random.choices(chars, k=length))
left = make_base36(random.randint(1, 6)).lstrip('0') or '0'
right = make_base36(random.randint(1, 6)).lstrip('0') or '0'
# Pad to at least 1 char
print(f'{left}:{right}')
""",

"cc_317_B. Ants": """\
import random
random.seed(42)
# n ants, t queries
n = 100
t = 10
print(n, t)
for _ in range(t):
    x = random.randint(0, 30)
    y = random.randint(0, x)
    print(x, y)
""",

"cc_388_C. Fox and Card Game": """\
import random
random.seed(42)
n = 5
print(n)
for _ in range(n):
    k = random.randint(2, 8)
    vals = [random.randint(100, 2000) for _ in range(k)]
    print(k, *vals)
""",

"cc_480_B. Long Jumps": """\
import random
random.seed(42)
l = 500
x = random.randint(10, 100)
y = random.randint(10, 100)
# n sorted positions including 0 and up to l
positions = sorted(random.sample(range(0, l+1), 20))
if 0 not in positions:
    positions[0] = 0
    positions.sort()
n = len(positions)
print(n, l, x, y)
print(*positions)
""",

"cc_1281_C. Cut and Paste": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    # Start with string of 2s so tape grows; x must be reachable
    init_len = random.randint(2, 4)
    x = init_len * 3  # x > init so tape grows; small enough to be fast
    s = '2' * init_len
    print(x)
    print(s)
""",

"cc_19_C. Deletion of Repeats": """\
import random
random.seed(42)
n = 50
print(n)
# values 0..99, allow duplicates
print(*[random.randint(0, 99) for _ in range(n)])
""",

"cc_343_C. Read Time": """\
import random
random.seed(42)
n, m = 8, 6
# h: n sorted positions (students), p: m sorted positions (problems)
h = sorted(random.sample(range(0, 10**9), n))
p = sorted(random.sample(range(0, 10**9), m))
print(n, m)
print(*h)
print(*p)
""",

"cc_366_E. Dima and Magic Guitar": """\
import random
random.seed(42)
n, m, k = 6, 6, 4
s_len = 5
print(n, m, k, s_len)
for _ in range(n):
    print(*[random.randint(1, k) for _ in range(m)])
print(*[random.randint(1, k) for _ in range(s_len)])
""",

"cc_38_D. Vasya the Architect": """\
import random
random.seed(42)
n = 8
print(n)
# Nested rectangles: each must contain the previous center
cx, cy = 0, 0
r = 50
for i in range(n):
    x1 = cx - r + random.randint(0, 5)
    y1 = cy - r + random.randint(0, 5)
    x2 = cx + r - random.randint(0, 5)
    y2 = cy + r - random.randint(0, 5)
    print(x1, y1, x2, y2)
    # next center is inside current rectangle
    cx = (x1 + x2) // 2 + random.randint(-5, 5)
    cy = (y1 + y2) // 2 + random.randint(-5, 5)
    r = r // 2 + 5
""",

"cc_438_B. The Child and Zoo": """\
import random
random.seed(42)
n, m = 15, 20
print(n, m)
print(*[random.randint(1, 10000) for _ in range(n)])
edges = set()
# connected: chain first
for i in range(1, n):
    print(i, i+1)
    edges.add((i, i+1))
added = n - 1
while added < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
        print(u, v)
        added += 1
""",

"cc_789_B. Masha and geometric depression": """\
import random
random.seed(42)
b1 = random.randint(1, 5)
q = random.randint(2, 4)  # |q|>1 → finite sequence
l = 1000
# Generate terms of b1*q^k until > l
terms = []
cur = b1
while abs(cur) <= l:
    terms.append(cur)
    cur *= q
m = len(terms) + 3
print(b1, q, l, m)
# a: some terms present, some not
present = random.sample(terms, min(2, len(terms)))
extra = [random.randint(-l, l) for _ in range(m - len(present))]
a = present + extra
random.shuffle(a)
print(*a)
""",

"cc_139_B. Wallpaper": """\
import random
random.seed(42)
n = 10
print(n)
for _ in range(n):
    a = random.randint(2, 20)  # width
    b = random.randint(2, 20)  # length
    c = random.randint(1, 10)  # height
    print(a, b, c)
m = 5
print(m)
for _ in range(m):
    length = random.randint(10, 100)  # roll length
    width = random.randint(1, 5)      # stripe width
    price = random.randint(10, 200)   # price
    print(length, width, price)
""",

"cc_106_D. Treasure Island": """\
import random
random.seed(42)
n, m = 8, 10
print(n, m)
grid = [['#']*m for _ in range(n)]
for i in range(1, n-1):
    for j in range(1, m-1):
        grid[i][j] = '.'
grid[2][3] = 'A'
grid[3][6] = 'B'
grid[5][2] = 'C'
for row in grid:
    print(''.join(row))
# Commands: "D steps" format (direction + space + number)
cmds = [('E', 2), ('S', 1), ('W', 1), ('N', 1)]
print(len(cmds))
for d, s in cmds:
    print(d, s)
""",

"cc_14_C. Four Segments": """\
# 4 line segments forming a rectangle
x1, y1, x2, y2 = 0, 0, 10, 8
# Bottom: (x1,y1)-(x2,y1), Top: (x1,y2)-(x2,y2)
# Left: (x1,y1)-(x1,y2), Right: (x2,y1)-(x2,y2)
print(x1, y1, x2, y1)
print(x2, y1, x2, y2)
print(x2, y2, x1, y2)
print(x1, y2, x1, y1)
""",

# ── batch 3 ──────────────────────────────────────────────────────────────────

"cc_1307_D. Cow and Fields": """\
import random
random.seed(42)
n, k = 30, 10
m = n - 1  # tree
print(n, m, k)
# k special nodes (1-indexed, distinct)
special = random.sample(range(1, n+1), k)
print(*special)
# tree edges
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

"cc_27_B. Tournament": """\
import random
random.seed(42)
n = 6
print(n)
# n*(n-1)/2 match results: each pair (i,j) — winner beats loser
teams = list(range(1, n+1))
for i in range(1, n+1):
    for j in range(i+1, n+1):
        # randomly assign winner
        if random.random() < 0.5:
            print(i, j)  # i beats j
        else:
            print(j, i)  # j beats i
""",

"cc_445_B. DZY Loves Chemistry": """\
import random
random.seed(42)
n, m = 30, 35
print(n, m)
# connected graph: chain first, then random edges
for i in range(1, n):
    print(i, i+1)
added = n - 1
edges = set((i, i+1) for i in range(1, n))
while added < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u,v))
        print(u, v)
        added += 1
""",

"cc_518_E. Arthur and Questions": """\
import random
random.seed(42)
n, k = 20, 4
print(n, k)
# Generate a valid period-k sequence, then replace some with '?'
# Within each period group, values must be strictly increasing
base = list(range(0, n))
tokens = []
for i in range(n):
    if random.random() < 0.3:
        tokens.append('?')
    else:
        tokens.append(str(base[i]))
print(' '.join(tokens))
""",

"cc_614_C. Peter and Snow Blower": """\
import random
random.seed(42)
n = 20
# center point not on the polygon
cx, cy = 0, 0
print(n, cx, cy)
# convex polygon vertices (roughly circular)
import math
for i in range(n):
    angle = 2 * math.pi * i / n
    r = random.randint(100, 200)
    x = int(r * math.cos(angle))
    y = int(r * math.sin(angle))
    print(x, y)
""",

"cc_1110_E. Magic Stones": """\
import random
random.seed(42)
n = 30
# Build a and b with the same sorted difference arrays (so answer is YES)
diffs = sorted([random.randint(1, 10) for _ in range(n-1)])
a = [random.randint(1, 100)]
for d in diffs:
    a.append(a[-1] + d)
b = [a[0]]  # same first element → YES
random.shuffle(diffs)
for d in diffs:
    b.append(b[-1] + d)
print(n)
print(*a)
print(*b)
""",

"cc_1500_B. Two chandeliers": """\
import random
random.seed(42)
n, m, k = 15, 20, 10
print(n, m, k)
# a: n distinct values, b: m values
a = random.sample(range(1, n*3), n)
b = [random.randint(1, n*3) for _ in range(m)]
print(*a)
print(*b)
""",

"cc_1438_C. Engineer Artem": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(3, 8)
    m = random.randint(3, 8)
    print(n, m)
    for i in range(n):
        print(*[random.randint(1, 20) for _ in range(m)])
""",

"cc_613_A. Peter and Snow Blower": """\
import random, math
random.seed(42)
n = 15
cx, cy = 0, 0
print(n, cx, cy)
for i in range(n):
    angle = 2 * math.pi * i / n
    r = random.randint(80, 150)
    x = int(r * math.cos(angle))
    y = int(r * math.sin(angle))
    print(x, y)
""",

"cc_1493_C. K-beautiful Strings": """\
import random, string
random.seed(42)
T = 5
print(T)
for _ in range(T):
    k = random.randint(2, 5)
    n = random.randint(k, 20)
    print(n, k)
    s = ''.join(random.choices(string.ascii_lowercase, k=n))
    print(s)
""",

"cc_474_C. Captain Marmot": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    # 4 squares, each with center (x,y) and two corner coords (a,b)
    for _ in range(4):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        a = random.randint(-100, 100)
        b = random.randint(-100, 100)
        print(x, y, a, b)
""",

"cc_967_D. Resource Distribution": """\
import random
random.seed(42)
n = 15
x = 100
y = 80
print(n, x, y)
print(*[random.randint(1, 100) for _ in range(n)])
""",

"cc_1165_E. Two Arrays and Sum of Functions": """\
import random
random.seed(42)
n = 50
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_1184_D1. Parallel Universes (Easy)": """\
import random
random.seed(42)
n, k, m, t = 10, 5, 20, 8
print(n, k, m, t)
for _ in range(t):
    a = random.randint(0, 1)
    b = random.randint(1, 10)
    print(a, b)
""",

"cc_703_C. Chris and Road": """\
import random
random.seed(42)
n = 20
w = 1000
v = random.randint(100, 900)
u = random.randint(100, 900)
print(n, w, v, u)
for _ in range(n):
    x = random.randint(1, 10000)
    y = random.randint(0, w-1)
    print(x, y)
""",

"cc_725_C. Hidden Word": """\
import random
random.seed(42)
# 27-char string: all 26 uppercase letters, one appears twice
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
dup = random.choice(letters)
s = letters + [dup]
random.shuffle(s)
print(''.join(s))
""",

"cc_815_B. Karen and Test": """\
import random
random.seed(42)
n = 20  # must be even for valid answer
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_860_C. Tests Renumeration": """\
import random, string
random.seed(42)
n = 10
print(n)
for i in range(n):
    name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    t = random.randint(0, 1)
    print(name, t)
""",

"cc_127_E. E-reader Display": """\
import random
random.seed(42)
n = 8
print(n)
for _ in range(n):
    print(''.join(random.choice('01') for _ in range(n)))
""",

"cc_163_A. Substring and Subsequence": """\
import random, string
random.seed(42)
n = 100
s = ''.join(random.choices(string.ascii_lowercase, k=n))
m = 50
s1 = ''.join(random.choices(string.ascii_lowercase, k=m))
print(s)
print(s1)
""",

# ── batch 2 ──────────────────────────────────────────────────────────────────

"cc_662_B. Graph Coloring": """\
import random
random.seed(42)
n, m = 20, 30
print(n, m)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    while v == u:
        v = random.randint(1, n)
    c = random.choice(['R', 'B'])
    print(u, v, c)
""",

"cc_79_C. Beaver": """\
import random, string
random.seed(42)
n_patterns = 8
s_len = 200
s = ''.join(random.choices(string.ascii_letters + string.digits + '_', k=s_len))
print(s)
print(n_patterns)
# generate substrings of s as patterns (guaranteed to appear in s)
for _ in range(n_patterns):
    start = random.randint(0, s_len - 10)
    length = random.randint(3, 10)
    print(s[start:start+length])
""",

"cc_914_E. Palindromes in a Tree": """\
import random, string
random.seed(42)
n = 50
print(n)
# random tree
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
print(''.join(random.choices(string.ascii_lowercase[:5], k=n)))
""",

"cc_1107_F. Vasya and Endless Credits": """\
import random
random.seed(42)
n = 20
print(n)
for _ in range(n):
    x = random.randint(1, 100)
    y = random.randint(1, 100)
    k = random.randint(1, 10)
    print(x, y, k)
""",

"cc_1382_C1. Prefix Flip (Easy Version)": """\
import random
random.seed(42)
T = 5
print(T)
for _ in range(T):
    n = random.randint(3, 20)
    a = ''.join(random.choice('01') for _ in range(n))
    b = ''.join(random.choice('01') for _ in range(n))
    print(n)
    print(a)
    print(b)
""",

"cc_1547_E. Air Conditioners": """\
import random
random.seed(42)
T = 4
print(T)
for _ in range(T):
    print()  # blank line before each test case
    n = random.randint(5, 20)
    k = random.randint(1, n)
    print(n, k)
    positions = sorted(random.sample(range(1, n+1), k))
    print(*positions)
    print(*[random.randint(1, 100) for _ in range(k)])
""",

"cc_339_C. Xenia and Weights": """\
import random
random.seed(42)
# 10-char binary string: which weights 1..10 are available
available = [random.randint(0, 1) for _ in range(10)]
# ensure at least some weights available
if not any(available):
    available[0] = 1
print(''.join(map(str, available)))
m = random.randint(5, 15)
print(m)
""",

"cc_500_E. New Year Domino": """\
import random
random.seed(42)
n = 20
print(n)
positions = sorted(random.sample(range(1, 10**6), n*2))
dominoes = [(positions[i*2], positions[i*2+1]) for i in range(n)]
for l, r in dominoes:
    print(l, r)
q = 10
print(q)
for _ in range(q):
    a = random.randint(1, n)
    b = random.randint(a, n)
    print(a, b)
""",

"cc_737_C. Subordinates": """\
import random
random.seed(42)
n = 20
s = random.randint(1, n)  # 1-indexed supervisor
print(n, s)
# t: n values each in 0..n-1 (employee ranks)
print(*[random.randint(0, n-1) for _ in range(n)])
""",

"cc_1056_D. Decorate Apple Tree": """\
import random
random.seed(42)
n = 30
print(n)
# parents: p[2]..p[n], each p[i] in 1..i-1
print(*[random.randint(1, i) for i in range(1, n)])
""",

"cc_1205_D. Almost All": """\
import random
random.seed(42)
n = 20
print(n)
# n-1 edges forming a tree
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

"cc_1288_D. Minimax Problem": """\
import random
random.seed(42)
n, m = 50, 4
print(n, m)
for _ in range(n):
    print(*[random.randint(0, 100) for _ in range(m)])
""",

"cc_1461_F. Mathematical Expression": """\
import random
random.seed(42)
n = 30
print(n)
# values: mix of 0s and non-zeros
a = [random.choice([0, random.randint(1, 10)]) for _ in range(n)]
print(*a)
# operations: mix of + and *
ops = ''.join(random.choice('+*') for _ in range(n-1))
print(ops)
""",

"cc_1538_C. Number of Pairs": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(10, 50)
    l = random.randint(1, 50)
    r = random.randint(l, 200)
    print(n, l, r)
    print(*[random.randint(1, 100) for _ in range(n)])
""",

"cc_231_E. Cactus": """\
import random
random.seed(42)
# Build a cactus: chain tree + back-edges forming triangles
n = 25
edges = [(i, i+1) for i in range(1, n)]
extra = []
used = set()
for i in range(3, n-1, 4):
    if i+2 <= n and i-1 not in used and i not in used:
        extra.append((i, i+2))
        used.add(i); used.add(i+1); used.add(i+2)
all_edges = edges + extra
m = len(all_edges)
print(n, m)
for u, v in all_edges:
    print(u, v)
# k queries (pairs of nodes)
k = 10
print(k)
for _ in range(k):
    a = random.randint(1, n)
    b = random.randint(1, n)
    print(a, b)
""",

"cc_351_B. Jeff and Furik": """\
import random
random.seed(42)
n = 50
print(n)
perm = list(range(1, n+1))
random.shuffle(perm)
print(*perm)
""",

"cc_514_E. Darth Vader and Tree": """\
import random
random.seed(42)
n = 50
x = 100  # edge weights must be <= 100 (used as array index N-d, N=101)
print(n, x)
print(*[random.randint(1, min(x, 100)) for _ in range(n)])
""",

"cc_631_D. Messenger": """\
import random, string
random.seed(42)
# tokens: "k-c" format, where k is count and c is a character
chars = string.ascii_lowercase[:6]
def make_tokens(length):
    tokens = []
    for _ in range(length):
        k = random.randint(1, 3)
        c = random.choice(chars)
        tokens.append(f'{k}-{c}')
    return tokens
n = 20; m = 8
print(n, m)
print(*make_tokens(n))
print(*make_tokens(m))
""",

"cc_1204_C. Anna, Svyatoslav and Maps": """\
import random
random.seed(42)
n = 12
print(n)
# adjacency matrix: symmetric binary strings
adj = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        v = random.randint(0, 1)
        adj[i][j] = adj[j][i] = v
for row in adj:
    print(''.join(map(str, row)))
m = random.randint(3, 8)
print(m)
# waypoints: m distinct 1-indexed nodes
print(*random.sample(range(1, n+1), m))
""",

"cc_1287_D. Numbers on Tree": """\
import random
random.seed(42)
n = 20
print(n)
# tree: node 1 is root (parent=0). Build random tree then assign valid c values.
children = [[] for _ in range(n+1)]
parent = [0] * (n+1)
parent[1] = 0
for i in range(2, n+1):
    p = random.randint(1, i-1)
    parent[i] = p
    children[p].append(i)
# For each node, c[i] = valid position in parent's children sorted list
# c[i] must be in [0, len(subtree of parent built so far)] — use 0 for safety
def subtree_size(v):
    return 1 + sum(subtree_size(c) for c in children[v])
for i in range(1, n+1):
    # c[i] can be at most len(result before inserting i) = sum of subtree sizes of siblings before i
    # simplest: c[i] = 0 always works
    c = 0
    print(parent[i], c)
""",

# ── next batch ───────────────────────────────────────────────────────────────

"cc_638_B. Making Genome in Berland": """\
# Chain of parts: "ab","bc","cd",... forms a valid genome "abcde..."
import random, string
random.seed(42)
chars = string.ascii_lowercase[:12]
n = len(chars) - 1
print(n)
for i in range(n):
    print(chars[i] + chars[i+1])
""",

"cc_1343_F. Restore the Permutation by Sorted Segments": """\
# Construct valid test: permutation 1..n with consecutive pair segments
# Each segment is {i, i+1} → ref reconstructs [1,2,...,n] from first=1
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(4, 8)
    print(n)
    perm = list(range(1, n+1))
    random.shuffle(perm)
    for i in range(1, n):
        seg = sorted(perm[:i+1])
        print(len(seg), *seg)
""",

"cc_1369_E. DeadLee": """\
import random
random.seed(42)
n, m = 50, 60
print(n, m)
# weights: each >= degree to guarantee ALIVE
w = [random.randint(2, 5) for _ in range(n)]
print(*w)
edges = set()
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        edges.add((u, v))
for u, v in edges:
    print(u, v)
""",

"cc_1506_G. Maximize the Remaining String": """\
import random, string
random.seed(42)
T = 5
print(T)
for _ in range(T):
    n = random.randint(10, 50)
    print(''.join(random.choices(string.ascii_lowercase[:6], k=n)))
""",

"cc_180_E. Cubes": """\
import random
random.seed(42)
n = 200
m = 10
k = 50
print(n, m, k)
print(*[random.randint(1, m) for _ in range(n)])
""",

"cc_228_E. The Road to Berland is Paved With Good Intentions": """\
import random
random.seed(42)
n, m = 80, 70
print(n, m)
nodes = list(range(1, n+1))
random.shuffle(nodes)
edges = set()
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        t = random.randint(0, 1)
        edges.add((u, v))
        print(u, v, t)
""",

"cc_488_D. Strip": """\
import random
random.seed(42)
n, l, s = 100, 3, 50
print(n, l, s)
# Generate sequence with small range so splitting works
base = random.randint(1, 200)
print(*[base + random.randint(0, s//2) for _ in range(n)])
""",

"cc_560_E. Gerald and Giant Chess": """\
import random
random.seed(42)
h, w, n = 1000, 1000, 50
print(h, w, n)
cells = set()
while len(cells) < n:
    r = random.randint(1, h)
    c = random.randint(1, w)
    if (r, c) != (1, 1) and (r, c) != (h, w):
        cells.add((r, c))
for r, c in sorted(cells):
    print(r, c)
""",

"cc_746_G. New Roads": """\
# a = levels list (increasing then decreasing); n = 1 + sum(a)
# k in [b[0], maxk]
a = [2, 3, 5, 4, 3]
t = len(a)
n = 1 + sum(a)
# compute b and maxk
b = [0] * t
b[t-1] = a[t-1]
maxk = a[t-1]
for i in range(t-2, -1, -1):
    b[i] = b[i+1]
    if a[i+1] < a[i]:
        b[i] += a[i] - a[i+1]
    maxk += a[i] - 1
k = (b[0] + maxk) // 2
print(n, t, k)
print(*a)
""",

"cc_814_D. An overnight dance in discotheque": """\
# Nested concentric circles: each circle contains all smaller-radius ones
n = 6
print(n)
for i in range(n):
    r = (i + 1) * 200
    print(0, 0, r)
""",

"cc_909_E. Coprocessor": """\
import random
random.seed(42)
n, m = 40, 50
print(n, m)
# types: 0 = CPU, 1 = coprocessor
types = [random.randint(0, 1) for _ in range(n)]
print(*types)
# DAG: edges only go from lower to higher index
edges = set()
while len(edges) < m:
    x = random.randint(0, n-2)
    y = random.randint(x+1, n-1)
    if (x, y) not in edges:
        edges.add((x, y))
        print(x, y)
""",

"cc_1311_B. WeirdSort": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(5, 15)
    # m = number of allowed adjacent swap positions (1..n-1)
    m = random.randint(1, n-1)
    positions = sorted(random.sample(range(1, n), m))
    print(n, m)
    print(*[random.randint(1, 10) for _ in range(n)])
    print(*positions)
""",

"cc_1420_C2. Pokémon Army (hard version)": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(5, 20)
    q = random.randint(3, 10)
    print(n, q)
    print(*[random.randint(0, 100) for _ in range(n)])
    for _ in range(q):
        l = random.randint(1, n)
        r = random.randint(1, n)
        print(l, r)
""",

"cc_1439_B. Graph Subset Problem": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    k = random.randint(2, 4)
    n = random.randint(k*2, 20)
    m = random.randint(k*n//2, min(n*(n-1)//2, 40))
    print(n, m, k)
    edges = set()
    while len(edges) < m:
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u != v and (u,v) not in edges and (v,u) not in edges:
            edges.add((u, v))
            print(u, v)
""",

"cc_1512_F. Education": """\
import random
random.seed(42)
T = 3
print(T)
for _ in range(T):
    n = random.randint(3, 10)
    c = random.randint(1, 20)
    print(n, c)
    print(*[random.randint(1, 10) for _ in range(n)])
    print(*[random.randint(1, 50) for _ in range(n)])
""",

"cc_167_B. Wizards and Huge Prize": """\
import random
random.seed(42)
n = 20
l = 10
a = 3  # number of -1s in s
print(n, l, a)
print(*[random.randint(0, 100) for _ in range(n)])
# s: a values are -1, rest are positive
s = [-1] * a + [random.randint(1, 5) for _ in range(n - a)]
random.shuffle(s)
print(*s)
""",

"cc_25_C. Roads in Berland": """\
import random
random.seed(42)
n = 8
print(n)
INF = 10**9
mat = [[INF]*n for _ in range(n)]
for i in range(n): mat[i][i] = 0
for i in range(n):
    for j in range(i+1, n):
        w = random.randint(1, 100)
        mat[i][j] = mat[j][i] = w
for row in mat:
    print(*row)
q = 10
print(q)
for _ in range(q):
    x = random.randint(1, n)
    y = random.randint(1, n)
    while y == x:
        y = random.randint(1, n)
    w = random.randint(1, 50)
    print(x, y, w)
""",

"cc_518_D. Ilya and Escalator": """\
import random
random.seed(42)
n = 50
p = round(random.uniform(0.1, 0.9), 2)
t = 100
print(n, p, t)
""",

"cc_544_D. Destroying Roads": """\
import random
random.seed(42)
n, m = 30, 50
print(n, m)
edges = set()
# ensure connected: chain
for i in range(1, n):
    print(i, i+1)
    edges.add((i, i+1))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u,v) not in edges and (v,u) not in edges:
        edges.add((u, v))
        print(u, v)
s1, t1, l1 = 1, n, random.randint(n-1, n+5)
s2, t2, l2 = random.randint(1, n//2), random.randint(n//2+1, n), random.randint(1, n)
print(s1, t1, l1)
print(s2, t2, l2)
""",

"cc_592_D. Super M": """\
import random
random.seed(42)
n = 30
m = 10
print(n, m)
# random tree
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
# m attacked cities (distinct)
attacked = random.sample(range(1, n+1), m)
print(*attacked)
""",

# ── large-input batch ────────────────────────────────────────────────────────

"cc_1227_D1. Optimal Subsequences (Easy Version)": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
m = 100000
print(m)
for _ in range(m):
    k = random.randint(1, n)
    pos = random.randint(1, k)
    print(k, pos)
""",

"cc_691_D. Swaps in Permutation": """\
import random
random.seed(42)
n = 100000
m = 200000
perm = list(range(1, n+1))
random.shuffle(perm)
print(n, m)
print(*perm)
for _ in range(m):
    a = random.randint(1, n)
    b = random.randint(1, n)
    print(a, b)
""",

"cc_316_B2. EKG": """\
import random
random.seed(42)
n = 200000
m = 200000
arr = [0] * n
arr[random.randint(0, n-1)] = 1
print(n, m)
print(*arr)
""",

"cc_765_D. Artsem and Saunders": """\
import random
random.seed(42)
n = 200000
# idempotent function: f(f(x)) = f(x)
# choose fixed points, map everything else to a fixed point
num_fixed = n // 2
fixed = random.sample(range(1, n+1), num_fixed)
fixed_set = set(fixed)
f = [0] * (n + 1)
for x in fixed:
    f[x] = x
non_fixed = [x for x in range(1, n+1) if x not in fixed_set]
for x in non_fixed:
    f[x] = random.choice(fixed)
print(n)
print(*f[1:])
""",

"cc_1525_D. Armchairs": """\
import random
random.seed(42)
n = 200000
# alternating 0 and 1
arr = [i % 2 for i in range(n)]
random.shuffle(arr)
print(n)
print(*arr)
""",

"cc_331_A2. Oh Sweet Beaverette": """\
import random
random.seed(42)
n = 200000
arr = [random.randint(-10**9, 10**9) for _ in range(n)]
print(n)
print(*arr)
""",

"cc_438_A. The Child and Toy": """\
import random
random.seed(42)
n = 100000
m = n - 1
print(n, m)
print(*[random.randint(1, 10**9) for _ in range(n)])
for i in range(2, n+1):
    print(random.randint(1, i-1), i)
""",

"cc_263_D. Cycle in Graph": """\
import random
random.seed(42)
n = 100000
m = 200000
k = 5
print(n, m, k)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    print(u, v)
""",

"cc_830_A. Office Keys": """\
import random
random.seed(42)
n = 100000
k = 100000
d = 2000000000
print(n, k, d)
keys = sorted(random.sample(range(-10**9, 10**9), n))
workers = sorted(random.sample(range(-10**9, 10**9), k))
print(*keys)
print(*workers)
""",

"cc_1315_D. Recommendations": """\
import random
random.seed(42)
n = 100000
print(n)
a = [random.randint(1, n) for _ in range(n)]
print(*a)
s = [random.randint(1, 10**9) for _ in range(n)]
print(*s)
""",

"cc_1475_D. Cleaning the Phone": """\
import random
random.seed(42)
t = 1
n = 200000
m = random.randint(n//4, n//2)
mem_needed = random.randint(10**8, 2*10**9)
print(t)
print(n, mem_needed)
a = [random.randint(1, 10**9) for _ in range(n)]
b = [random.choice([1, 2]) for _ in range(n)]
print(*a)
print(*b)
""",

"cc_1016_C. Vasya And The Mushrooms": """\
import random
random.seed(42)
n = 300000
print(n)
print(*[random.randint(0, 10**9) for _ in range(n)])
print(*[random.randint(0, 10**9) for _ in range(n)])
""",

"cc_1081_D. Maximum Distance": """\
import random
random.seed(42)
n = 100000
m = 100000
q = 100000
print(n, m, q)
print(*[random.randint(1, 10**9) for _ in range(n)])
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    w = random.randint(1, 10**9)
    print(u, v, w)
for _ in range(q):
    print(random.randint(1, m))
""",

"cc_191_A. Dynasty Puzzles": """\
import random, string
random.seed(42)
n = 100000
print(n)
vowels = set('aeiou')
chars = string.ascii_lowercase
for _ in range(n):
    length = random.randint(1, 10)
    print(''.join(random.choices(chars, k=length)))
""",

"cc_940_B. Our Tanya is Crying Out Loud": """\
import random
random.seed(42)
n = 2000000000
m = random.randint(1, n)
r = random.randint(1, n)
b = random.randint(1, n)
print(n)
print(m)
print(r)
print(b)
""",

"cc_830_C. Bamboo Partition": """\
import random
random.seed(42)
n = 100000
k = 10**9
print(n, k)
print(*sorted(random.randint(1, 10**9) for _ in range(n)))
""",

"cc_61_E. Enemy is weak": """\
import random
random.seed(42)
n = 200000
# random permutation of 1..n (no duplicates, all in valid range)
perm = random.sample(range(1, n+1), n)
print(n)
print(*perm)
""",

"cc_1153_D. Serval and Rooted Tree": """\
import random
random.seed(42)
n = 200000
node_type = [random.randint(0, 1) for _ in range(n)]
print(n)
print(*node_type)
# tree edges: parent of i (for i=2..n)
for i in range(2, n+1):
    print(random.randint(1, i-1))
""",

"cc_1400_D. Zigzags": """\
import random
random.seed(42)
t = 1
n = 200000
print(t)
print(n)
print(*[random.randint(1, n) for _ in range(n)])
""",

"cc_466_C. Number of Ways": """\
import random
random.seed(42)
n = 200000
# array with sum 0, elements in {-1, 0, 1}
arr = []
total = 0
for i in range(n-1):
    v = random.choice([-1, 0, 1])
    arr.append(v)
    total += v
arr.append(-total)  # ensure sum = 0
print(n)
print(*arr)
""",

"cc_1041_C. Coffee Break": """\
import random
random.seed(42)
n = 200000
m = 300000
d = random.randint(1, 300)
print(n, m, d)
print(*sorted(random.randint(1, m) for _ in range(n)))
""",

"cc_1009_D. Relatively Prime Graph": """\
print(1000, 500000)
""",

"cc_22_C. System Administrator": """\
print(300, 44850, 150)
""",

"cc_1519_C. Berland Regional": """\
import random
random.seed(42)
t = 1
n = 200000
r = random.randint(1, 10)
print(t)
print(n)
print(*[random.randint(1, r) for _ in range(n)])
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_1290_A. Mind Control": """\
import random
random.seed(42)
t = 1
n = 200000
m = random.randint(1, n)
k = random.randint(0, m-1)
print(t)
print(n, m, k)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_724_D. Dense Subsequence": """\
import random, string
random.seed(42)
n = 200000
print(n)
print(''.join(random.choices(string.ascii_lowercase, k=n)))
""",

"cc_85_B. Embassy Queue": """\
import random
random.seed(42)
n = 100000
m = 200000
k = 50000
print(n, m, k)
# n+1 timestamps for n workers
timestamps = sorted(random.randint(1, 10**9) for _ in range(n+1))
print(*timestamps)
print(m)
queries = sorted(random.randint(1, 10**12) for _ in range(m))
print(*queries)
""",

"cc_453_C. Little Pony and Summer Sun Celebration": """\
import random
random.seed(42)
n = 100000
m = 100000
print(n, m)
# random tree
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
# 0/1 array
print(*[random.randint(0, 1) for _ in range(n)])
""",

"cc_1106_D. Lunar New Year and a Wander": """\
import random
random.seed(42)
n = 200000
m = 200000
print(n, m)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    while u == v:
        v = random.randint(1, n)
    print(u, v)
""",

"cc_1095_F. Make It Connected": """\
import random
random.seed(42)
n = 100000
m = 1
print(n, m)
print(*[random.randint(1, 10**9) for _ in range(n)])
print(*random.sample(range(1, n+1), m))
""",

"cc_1434_A. Perform Easily": """\
import random
random.seed(42)
print(*[random.randint(1, 10**9) for _ in range(6)])
m = 200000
print(m)
for _ in range(m):
    print(*[random.randint(1, 10**9) for _ in range(6)])
""",

"cc_626_D. Jerry's Protest": """\
import random
random.seed(42)
n = 100000
print(n)
print(*random.sample(range(1, 2*10**9), n))
""",

"cc_242_C. King's Path": """\
import random
random.seed(42)
# unreachable destination far away, many forbidden strips
x0, y0 = 1, 1
x1, y1 = 10**9, 10**9
n = 100000
print(x0, y0, x1, y1)
print(n)
xs = random.sample(range(2, 10**9), n)
for x in xs:
    y1s = random.randint(1, 5*10**8)
    y2s = random.randint(y1s, min(y1s + 100, 10**9))
    print(x, y1s, y2s)
""",

"cc_984_D. XOR-pyramid": """\
import random
random.seed(42)
n = 5000
print(n)
print(*[random.randint(0, 2**20-1) for _ in range(n)])
m = 100000
print(m)
for _ in range(m):
    l = random.randint(1, n)
    r = random.randint(l, n)
    print(l, r)
""",

"cc_1468_J. Road Reform": """\
import random
random.seed(42)
t = 1
n = 100000
m = n - 1
k = random.randint(1, 10**9)
print(t)
print(n, m, k)
# random spanning tree
for i in range(2, n+1):
    p = random.randint(1, i-1)
    w = random.randint(1, 10**9)
    print(p, i, w)
""",

"cc_523_D. Statistics of Recompressing Videos": """\
import random
random.seed(42)
n = 200000
m = 1
print(n, m)
for _ in range(n):
    a = random.randint(10**8, 10**9)
    b = random.randint(a, a + 10**8)
    print(a, b)
print(random.randint(1, 10**9))
""",

"cc_793_D. Presents in Bankopolis": """\
import random
random.seed(42)
n = 100000
m = 100000
b = 10**9
print(n, m)
print(b)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    while u == v:
        v = random.randint(1, n)
    w = random.randint(1, 10**9)
    print(u, v, w)
""",

"cc_327_D. Block Tower": """\
import random
random.seed(42)
n = 1000
m = 1000
print(n, m)
for _ in range(n):
    row = ''.join(random.choices('.#', weights=[3, 1], k=m))
    print(row)
""",

"cc_28_C. Bath Queue": """\
import random
random.seed(42)
n = 50
m = 50
print(n, m)
print(*[random.randint(1, 50) for _ in range(n)])
print(*[random.randint(1, 50) for _ in range(m)])
""",

"cc_977_D. Divide by three, multiply by two": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_1287_D. Numbers on Tree": """\
import random
random.seed(42)
n = 100000
print(n)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    t = random.randint(0, 1)
    k = random.randint(0, 3) if t == 1 else 0
    print(p, t, k)
""",

"cc_1205_D. Almost All": """\
import random
random.seed(42)
n = 100000
print(n)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

"cc_445_B. DZY Loves Chemistry": """\
import random
random.seed(42)
n = 100000
m = n - 1
print(n, m)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",

"cc_378_C. Maze": """\
import random
random.seed(42)
n = 1000
m = 1000
k = random.randint(1, 200000)
print(n, m, k)
for _ in range(n):
    row = ''.join(random.choices('.#', weights=[4, 1], k=m))
    print(row)
""",

"cc_377_A. Maze": """\
import random
random.seed(42)
n = 1000
m = 1000
k = random.randint(1, n*m-2)
print(n, m, k)
for _ in range(n):
    row = ''.join(random.choices('.#', weights=[4, 1], k=m))
    print(row)
""",

"cc_1381_A1. Prefix Flip (Easy Version)": """\
import random
random.seed(42)
t = 1
n = 200000
print(t)
print(n)
a = ''.join(random.choices('01', k=n))
b = ''.join(random.choices('01', k=n))
print(a)
print(b)
""",

"cc_681_D. Gifts by the List": """\
import random
random.seed(42)
n = 100000
m = n - 1
print(n, m)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
# each node's gift: a node in its subtree (just use the node itself)
print(*range(1, n+1))
""",

"cc_1294_D. MEX maximizing": """\
import random
random.seed(42)
n = 100000
k = random.randint(1, 10**9)
print(n, k)
for _ in range(n):
    print(random.randint(0, 10**9))
""",

"cc_1334_D. Minimum Euler Cycle": """\
import random
random.seed(42)
n = 100000
# Build Eulerian directed graph: for each node, out_deg = in_deg
# Simplest: chain 1->2->3->...->n->1
# Then add extra edges as pairs (u,v) and (v,u) to keep degrees balanced
print(n)
# adjacency: node i points to (i%n)+1 (cycle)
adj = [[] for _ in range(n+1)]
adj[n].append(1)
for i in range(1, n):
    adj[i].append(i+1)
# add random balanced pairs
for _ in range(n//2):
    u = random.randint(1, n)
    v = random.randint(1, n)
    adj[u].append(v)
    adj[v].append(u)
for i in range(1, n+1):
    print(len(adj[i]), *adj[i])
""",

"cc_1107_D. Compression": """\
import random
random.seed(42)
# n must be a perfect square (grid of sqrt(n) x sqrt(n) hex pixels)
import math
side = 50  # 50x50 = 2500
n = side * side
print(n)
hex_chars = '0123456789ABCDEF'
for _ in range(n):
    print(''.join(random.choices(hex_chars, k=4)))
""",

"cc_1547_E. Air Conditioners": """\
import random
random.seed(42)
t = 1
n = 200000
m = n // 2
print(t)
print(n, m)
# n positions
pos = sorted(random.sample(range(1, n+1), m))
temps = [random.randint(1, 10**9) for _ in range(m)]
for p, tp in zip(pos, temps):
    print(p, tp)
""",

"cc_1307_D. Cow and Fields": """\
import random
random.seed(42)
n = 100000
m = n - 1
k = min(n, 100000)
print(n, m, k)
cows = random.sample(range(1, n+1), k)
print(*cows)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    print(p, i)
""",


# ── correct large-n batch (round 2) ──────────────────────────────────────────

"cc_263_D. Cycle in Graph": """\
import random
random.seed(42)
n = 100000
m = 100000
k = 3
print(n, m, k)
seen = set()
# ensure a k-cycle exists
for i in range(1, k):
    print(i, i + 1)
    seen.add((i, i + 1))
print(k, 1)
seen.add((k, 1))
while len(seen) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u, v) not in seen:
        seen.add((u, v))
        print(u, v)
""",

"cc_940_B. Our Tanya is Crying Out Loud": """\
import random
random.seed(42)
n = 10 ** 9
k = random.randint(1, n)
A = random.randint(1, 10 ** 9)
B = random.randint(1, 10 ** 9)
print(n)
print(k)
print(A)
print(B)
""",

"cc_1016_C. Vasya And The Mushrooms": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.randint(0, 10 ** 9) for _ in range(n)])
print(*[random.randint(0, 10 ** 9) for _ in range(n)])
""",

"cc_453_C. Little Pony and Summer Sun Celebration": """\
import random
random.seed(42)
N = 100000
M = 100000
print(N, M)
edges = set()
for i in range(1, N):
    edges.add((i, i + 1))
    print(i, i + 1)
while len(edges) < M:
    u = random.randint(1, N)
    v = random.randint(1, N)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        edges.add((u, v))
        print(u, v)
print(*[random.randint(1, N) for _ in range(N)])
""",

"cc_61_E. Enemy is weak": """\
import random
random.seed(42)
n = 100000
perm = random.sample(range(1, n + 1), n)
print(n)
print(*perm)
""",

"cc_1153_D. Serval and Rooted Tree": """\
import random
random.seed(42)
n = 200000
print(n)
for i in range(2, n + 1):
    print(random.randint(1, i - 1))
""",

"cc_1041_C. Coffee Break": """\
import random
random.seed(42)
n = 100000
m = 50000
d = 1
print(n, m, d)
print(*sorted(random.randint(1, m) for _ in range(n)))
""",

"cc_681_D. Gifts by the List": """\
import random
random.seed(42)
n = 100000
m = 100000
print(n, m)
for _ in range(m):
    p = random.randint(1, n)
    c = random.randint(1, n)
    print(p, c)
print(*[random.randint(1, n) for _ in range(n)])
""",

"cc_1106_D. Lunar New Year and a Wander": """\
import random
random.seed(42)
n = 100000
m = 100000
print(n, m)
edges = set()
for i in range(1, n):
    edges.add((i, i + 1))
    print(i, i + 1)
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        edges.add((u, v))
        print(u, v)
""",

"cc_1294_D. MEX maximizing": """\
import random
random.seed(42)
q = 400000
x = 1
print(q, x)
for _ in range(q):
    print(random.randint(0, 10 ** 6))
""",

"cc_1315_D. Recommendations": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(1, n) for _ in range(n)])
print(*[random.randint(1, 10 ** 9) for _ in range(n)])
""",

"cc_1451_C. String Equality": """\
import random, string
random.seed(42)
t = 1
n = 200000
k = 10
print(t)
print(n, k)
a = ''.join(random.choices(string.ascii_lowercase[:5], k=n))
b = sorted(a)
random.shuffle(b)
print(a)
print(''.join(b))
""",

"cc_1396_C. Monster Invaders": """\
import random
random.seed(42)
N = 100000
a = random.randint(1, 10 ** 9)
b = random.randint(1, 10 ** 9)
c = random.randint(1, 10 ** 9)
k = random.randint(1, 10 ** 9)
print(N, a, b, c, k)
print(*[random.randint(1, 10 ** 9) for _ in range(N)])
""",

"cc_724_D. Dense Subsequence": """\
import random, string
random.seed(42)
m = 5
s = ''.join(random.choices(string.ascii_lowercase, k=100000))
print(m)
print(s)
""",

"cc_1075_D. Intersecting Subtrees": """\
import random
random.seed(42)
t = 1
n = 100000
print(t)
print(n)
for i in range(2, n + 1):
    p = random.randint(1, i - 1)
    print(p, i)
""",

"cc_1400_D. Zigzags": """\
import random
random.seed(42)
t = 1
n = 500
print(t)
print(n)
print(*[random.randint(1, n) for _ in range(n)])
""",

"cc_984_D. XOR-pyramid": """\
import random
random.seed(42)
n = 5000
print(n)
print(*[random.randint(0, 10 ** 9) for _ in range(n)])
""",

"cc_1468_J. Road Reform": """\
import random
random.seed(42)
t = 1
n = 5000
m = 8000
k = random.randint(1, 10 ** 9)
print(t)
print(n, m, k)
edges = set()
for i in range(1, n):
    w = random.randint(1, 10 ** 9)
    edges.add((i, i + 1))
    print(i, i + 1, w)
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v and (u, v) not in edges and (v, u) not in edges:
        edges.add((u, v))
        w = random.randint(1, 10 ** 9)
        print(u, v, w)
""",

"cc_102_D. Buses": """\
import random
random.seed(42)
a = 100000
b = 100000
print(a, b)
for _ in range(b):
    x = random.randint(1, a)
    y = random.randint(x, a)
    print(x, y)
""",

"cc_1422_D. Returning Home": """\
import random
random.seed(42)
n = 1000
m = 50
sx, sy = 1, 1
fx, fy = n, n
print(n, m)
print(sx, sy, fx, fy)
for _ in range(m):
    x = random.randint(1, n)
    y = random.randint(1, n)
    print(x, y)
for i in range(n):
    row = [str(random.randint(1, 10 ** 9)) for _ in range(n)]
    print(' '.join(row))
""",

"cc_793_D. Presents in Bankopolis": """\
import random
random.seed(42)
N = 1000
M = 1000
print(N)
edge_list = []
for i in range(1, N + 1):
    for j in range(i + 1, min(i + 3, N + 1)):
        edge_list.append((i, j, random.randint(1, 10 ** 9)))
print(len(edge_list))
for e in edge_list:
    print(*e)
print(M)
for _ in range(M):
    print(random.randint(1, N))
""",

"cc_1519_C. Berland Regional": """\
import random
random.seed(42)
T = 1
n = 100000
p = 100
print(T)
print(n)
print(*[random.randint(1, p) for _ in range(n)])
print(*[random.randint(1, 10 ** 6) for _ in range(n)])
""",

}


# ── Auto-generator ────────────────────────────────────────────────────────────

def _extract_max_n(description: str) -> int:
    """Best-effort: find the largest n upper-bound mentioned in the description."""
    candidates = []

    # Handle power notation: 10^5, 10^6, 2*10^5, 2×10^5, 2·10^5, 10**5
    for m in re.finditer(
        r'(\d+)\s*[×x\*·]?\s*10\s*[\^]\s*(\d+)|(\d+)\s*\*\*\s*(\d+)',
        description
    ):
        try:
            if m.group(1) is not None:
                val = int(m.group(1)) * (10 ** int(m.group(2)))
            else:
                val = int(m.group(3)) ** int(m.group(4))
            if 100 <= val <= 10**9:
                candidates.append(val)
        except Exception:
            pass

    # Superscript digits after 10: 10⁵ 10⁶
    sup_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    for m in re.finditer(r'10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', description):
        try:
            val = 10 ** int(m.group(1).translate(sup_map))
            if 100 <= val <= 10**9:
                candidates.append(val)
        except Exception:
            pass

    # Plain integers after ≤ or < (handles "≤ 200 000" with spaces)
    for m in re.finditer(r'[≤<]\s*(\d[\d\s]{0,10})', description):
        try:
            val = int(m.group(1).replace(' ', ''))
            if 100 <= val <= 10**9:
                candidates.append(val)
        except ValueError:
            pass

    if not candidates:
        return 100000  # safe default — better than 1000

    # Prefer the largest value that's a useful stress-test size
    for v in sorted(candidates, reverse=True):
        if v >= 1000:
            return min(v, 200000)
    return min(max(candidates), 200000)


def _auto_generator(problem: dict) -> str:
    """
    Build a generator by scaling up the problem's existing stdin.
    Detects common patterns: single-int, n+array, n+m+grid, t-test-cases, etc.
    """
    stdin  = problem["stdin"].strip()
    desc   = problem["description"]
    lines  = stdin.splitlines()
    max_n  = _extract_max_n(desc)

    if not lines:
        return ""

    def tok(line):
        return line.strip().split()

    t0 = tok(lines[0])

    # ── single line input ────────────────────────────────────────────────────
    if len(lines) == 1:
        if len(t0) == 1 and t0[0].lstrip('-').isdigit():
            n = min(max_n, 10**9)
            return f"print({n})\n"
        if all(x.lstrip('-').isdigit() for x in t0):
            vals = [min(int(x), 10**9) if int(x) > 0 else max(int(x), -10**9)
                    for x in t0]
            return f"print({', '.join(str(v) for v in vals)})\n"
        # string input
        n = min(max_n, 100000)
        import random
        return (f"import random, string\nrandom.seed(42)\n"
                f"print(''.join(random.choices(string.ascii_lowercase, k={n})))\n")

    # ── detect t test cases (first line is small int, next line starts a block) ─
    try:
        t_val = int(t0[0]) if len(t0) == 1 else -1
    except ValueError:
        t_val = -1

    if 1 < t_val <= 100 and len(lines) > 1:
        # grab one test case block (lines after line 0 up to t_val blocks)
        block_lines = lines[1:]
        # try to detect block size
        # look at line 1: if it's a single int → n, then n lines follow
        t1 = tok(block_lines[0]) if block_lines else []
        try:
            inner_n = int(t1[0]) if len(t1) == 1 else -1
        except (ValueError, IndexError):
            inner_n = -1

        if inner_n > 0 and len(block_lines) > inner_n:
            # pattern: t\nn\n[n lines]
            # reproduce with 1 test case, scaled n
            n_new = min(max_n, 50000)
            sample_inner = block_lines[1] if len(block_lines) > 1 else ""
            ti = tok(sample_inner)
            # single array line
            if all(x.lstrip('-').isdigit() for x in ti):
                return (f"import random\nrandom.seed(42)\n"
                        f"print(1)\nprint({n_new})\n"
                        f"print(*[random.randint(1, 10**9) for _ in range({n_new})])\n")
            # string line
            return (f"import random, string\nrandom.seed(42)\n"
                    f"print(1)\nprint({n_new})\n"
                    f"print(''.join(random.choices(string.ascii_lowercase, k={n_new})))\n")

        if len(t1) == 2 and all(x.lstrip('-').isdigit() for x in t1):
            # pattern: t\nn m\n...
            n_new = min(max_n, 50000)
            return (f"import random\nrandom.seed(42)\n"
                    f"print(1)\nprint({n_new}, {n_new})\n"
                    f"print(*[random.randint(1, 10**9) for _ in range({n_new})])\n")

        # fallback: t test cases, each a single int
        t_new = min(t_val * 10, 1000)
        n_new = min(max_n, 10**9)
        return (f"print({t_new})\n"
                f"for _ in range({t_new}):\n"
                f"    print({n_new})\n")

    # ── n on first line, then one array line ─────────────────────────────────
    if len(t0) == 1 and t0[0].isdigit() and len(lines) >= 2:
        n_new = min(max_n, 50000)
        t1 = tok(lines[1])
        if all(x.lstrip('-').isdigit() for x in t1):
            return (f"import random\nrandom.seed(42)\n"
                    f"n = {n_new}\nprint(n)\n"
                    f"print(*[random.randint(1, 10**9) for _ in range(n)])\n")
        # string second line
        if t1 and not t1[0].lstrip('-').isdigit():
            return (f"import random, string\nrandom.seed(42)\n"
                    f"n = {n_new}\nprint(n)\n"
                    f"print(''.join(random.choices(string.ascii_lowercase, k=n)))\n")

    # ── n m on first line ────────────────────────────────────────────────────
    if len(t0) == 2 and all(x.lstrip('-').isdigit() for x in t0):
        n_new = min(max_n, 50000)
        t1 = tok(lines[1]) if len(lines) > 1 else []
        if all(x.lstrip('-').isdigit() for x in t1):
            return (f"import random\nrandom.seed(42)\n"
                    f"n, m = {n_new}, {n_new}\nprint(n, m)\n"
                    f"print(*[random.randint(1, 10**9) for _ in range(n)])\n")
        # grid
        if t1 and len(t1[0]) > 1:
            return (f"import random, string\nrandom.seed(42)\n"
                    f"n, m = 500, 500\nprint(n, m)\n"
                    f"for _ in range(n):\n"
                    f"    print(''.join(random.choices(string.ascii_lowercase, k=m)))\n")

    # ── fallback: scale up first line integers ───────────────────────────────
    if all(x.lstrip('-').isdigit() for x in t0):
        vals = []
        for x in t0:
            v = int(x)
            if v > 0:
                vals.append(min(v * 100, min(max_n, 10**9)))
            else:
                vals.append(max(v * 100, -10**9))
        return f"print({', '.join(str(v) for v in vals)})\n"

    return ""


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_script(code: str, stdin: str = "", timeout: int = 30):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["python3", path],
            input=stdin.encode(),
            capture_output=True,
            timeout=timeout,
        )
        return r.stdout.decode(errors="replace"), r.returncode
    except subprocess.TimeoutExpired:
        return "", 1
    except Exception:
        return "", 1
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _run_script_file_stdin(code: str, stdin: str = "", timeout: int = 60):
    """Like _run_script but feeds stdin via a real file (fixes os.fstat tricks)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fc:
        fc.write(code)
        code_path = fc.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fi:
        fi.write(stdin)
        stdin_path = fi.name
    try:
        with open(stdin_path, "rb") as stdin_file:
            r = subprocess.run(
                ["python3", code_path],
                stdin=stdin_file,
                capture_output=True,
                timeout=timeout,
            )
        return r.stdout.decode(errors="replace"), r.returncode
    except subprocess.TimeoutExpired:
        return "", 1
    except Exception:
        return "", 1
    finally:
        for p in (code_path, stdin_path):
            try:
                os.unlink(p)
            except Exception:
                pass


def _test_one(gen_code: str, ref_code: str) -> tuple[bool, str]:
    gen_out, rc = _run_script(gen_code)
    if rc != 0 or not gen_out.strip():
        return False, f"gen rc={rc}"
    # Use file-based stdin so os.fstat(0).st_size returns the real size
    ref_out, rrc = _run_script_file_stdin(ref_code, stdin=gen_out)
    if rrc != 0 or not ref_out.strip():
        return False, f"ref rc={rrc}"
    return True, ""


def _save_generator(task_id: str, gen_code: str):
    with open(TRAIN_CACHE) as f:
        data = json.load(f)
    for p in data["problems"]:
        if p["task_id"] == task_id:
            p["test_case_generator"] = gen_code
            break
    with open(TRAIN_CACHE, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max problems to process (0 = all)")
    args = parser.parse_args()

    with open(TRAIN_CACHE) as f:
        data = json.load(f)

    todo = [p for p in data["problems"] if not p.get("test_case_generator")]
    print(f"[gen] {len(todo)} problems need generators")

    passed = failed = skipped = 0
    limit = args.limit or len(todo)

    for i, problem in enumerate(todo[:limit]):
        tid = problem["task_id"]
        gen_code = MANUAL.get(tid) or _auto_generator(problem)

        if not gen_code.strip():
            print(f"  SKIP  [{i+1}] {tid} — no generator")
            skipped += 1
            continue

        ok, reason = _test_one(gen_code, problem["ref_solution"])
        if not ok:
            ok, reason = _test_one(gen_code, problem["ref_solution"])  # one retry

        if ok:
            _save_generator(tid, gen_code)
            passed += 1
            print(f"  PASS  [{i+1}] {tid}")
        else:
            failed += 1
            print(f"  FAIL  [{i+1}] {tid} — {reason}")

        if (i + 1) % 25 == 0:
            print(f"  --- progress: {i+1}/{limit} | "
                  f"passed={passed} failed={failed} skipped={skipped}")

    print(f"\n[gen] Done. passed={passed} failed={failed} skipped={skipped}")


if __name__ == "__main__":
    main()
