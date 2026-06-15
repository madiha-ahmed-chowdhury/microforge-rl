#!/usr/bin/env python3
"""
generate_stress_tests_eval.py

Hand-crafted stress-test generators for the 60 evaluation problems in
online_rl/cc_pool_cache_test.json.

Usage:
    # Verify generators and embed them into the cache:
    python3 online_rl/generate_stress_tests_eval.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_PATH = "online_rl/cc_pool_cache_test.json"

GENERATORS = {

# ── HIGH MEMORY ───────────────────────────────────────────────────────────────

"cc_1037_E. Trips": """\
import random
random.seed(42)
n = 100000
m = 100000
k = 50
edges = set()
while len(edges) < m:
    a = random.randint(1, n)
    b = random.randint(1, n)
    if a != b:
        edges.add((min(a,b), max(a,b)))
print(n, m, k)
for a, b in list(edges)[:m]:
    print(a, b)
""",

"cc_1208_D. Restore Permutation": """\
import random
random.seed(42)
n = 100000
perm = list(range(1, n+1))
random.shuffle(perm)
# Compute s[i] = sum of p[j] for j<i where p[j]<p[i], using Fenwick tree
bit = [0] * (n + 2)
def update(i, v):
    while i <= n:
        bit[i] += v
        i += i & (-i)
def query(i):
    s = 0
    while i > 0:
        s += bit[i]
        i -= i & (-i)
    return s
s = []
for x in perm:
    s.append(query(x - 1))
    update(x, x)
print(n)
print(*s)
""",

"cc_1269_E. K Integers": """\
import random
random.seed(42)
n = 200000
perm = list(range(1, n+1))
random.shuffle(perm)
print(n)
print(*perm)
""",

"cc_1291_E. Prefix Enlightenment": """\
import random
random.seed(42)
n = 2000
k = 2000
state = ''.join(random.choice('01') for _ in range(n))
print(n, k)
print(state)
# Each lamp in at most 2 subsets
in_count = [0] * (n + 1)
subsets = []
for _ in range(k):
    members = []
    for lmp in random.sample(range(1, n+1), min(10, n)):
        if in_count[lmp] < 2:
            members.append(lmp)
            in_count[lmp] += 1
        if len(members) >= 5:
            break
    if not members:
        members = [1]
    subsets.append(members)
for s in subsets:
    print(len(s))
    print(*s)
q = 50000
print(q)
for _ in range(q):
    print(random.randint(1, k))
""",

"cc_1311_F. Moving Points": """\
import random
random.seed(42)
n = 200000
xs = sorted(random.sample(range(-10**8, 10**8), n))
vs = [random.randint(-10**8, 10**8) for _ in range(n)]
print(n)
print(*xs)
print(*vs)
""",

"cc_1354_F. Summoning Minions": """\
import random
random.seed(42)
t = 1
print(t)
n = 200000
k = 100
print(n, k)
for _ in range(n):
    print(random.randint(0, 10**6), random.randint(0, 10**6))
""",

"cc_1466_F. Euclid's nightmare": """\
import random
random.seed(42)
n = 50000
m = 100000
print(n, m)
for _ in range(n):
    size = random.randint(1, min(5, m))
    members = sorted(random.sample(range(1, m+1), size))
    print(len(members), *members)
""",

"cc_1490_G. Old Floppy Drive ": """\
import random
random.seed(42)
t = 1
print(t)
n = 100000
m = 1000
arr = [random.randint(-10**9, 10**9) for _ in range(n)]
queries = [random.randint(1, 10**15) for _ in range(m)]
print(n, m)
print(*arr)
print(*queries)
""",

"cc_1540_C1. Converging Array (Easy Version)": """\
import random
random.seed(42)
n = 200
# C: sorted non-decreasing array of n values
C = sorted([random.randint(1, 100) for _ in range(n)])
# B: n-1 minimum difference bounds (small non-negative values)
B = [random.randint(0, 2) for _ in range(n - 1)]
x = random.randint(0, 50)   # the single query value
print(n)
print(*C)
print(*B)
print(1)   # q = 1 in easy version
print(x)
""",

"cc_730_J. Bottles": """\
import random
random.seed(42)
n = 100
a = [random.randint(1, 100) for _ in range(n)]
b = [a[i] + random.randint(0, 100) for i in range(n)]
print(n)
print(*a)
print(*b)
""",

"cc_846_E. Chemistry in Berland": """\
import random
random.seed(42)
n = 200000
a = [random.randint(1, 10**9) for _ in range(n)]
b = [a[i] + random.randint(0, 10**8) for i in range(n)]
print(n)
print(*b)
print(*a)
# n-1 lines: 1-indexed parent, integer edge weight (1..100)
for i in range(1, n):
    parent = random.randint(1, i)
    w = random.randint(1, 100)
    print(parent, w)
""",

"cc_893_D. Credit Card": """\
import random
random.seed(42)
n = 100000
lim = 10**9
print(n, lim)
vals = []
for _ in range(n):
    r = random.random()
    if r < 0.4:
        vals.append(random.randint(1, 10**6))
    elif r < 0.7:
        vals.append(-random.randint(1, 10**6))
    else:
        vals.append(0)
print(*vals)
""",

"cc_963_B. Destruction of a Tree": """\
import random
random.seed(42)
# n must be odd; ref reads ALL parents on one line: p[i] = parent of node i+1 (0 means root)
n = 499999
parents = [0]  # node 1 is root
for i in range(2, n + 1):
    parents.append(random.randint(1, i - 1))
print(n)
print(*parents)
""",

"cc_1176_F. Destroy it!": """\
import random
random.seed(42)
n = 5000
print(n)
for turn in range(n):
    m = random.randint(1, 5)
    print(m)
    for _ in range(m):
        c = random.randint(1, 3)   # card class must be 1, 2, or 3
        d = random.randint(1, 10**9)
        print(c, d)
""",

"cc_1277_E. Two Fairs": """\
import random
random.seed(42)
t = 1
print(t)
n = 100000
m = 200000
a, b = 1, 2
print(n, m, a, b)
edges = set()
for i in range(1, n):
    edges.add((i, i+1))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v:
        edges.add((min(u,v), max(u,v)))
for u, v in list(edges)[:m]:
    print(u, v)
""",

"cc_1497_D. Genius": """\
import random
random.seed(42)
t = 3
print(t)
for _ in range(t):
    n = 20
    print(n)
    tags   = [random.randint(0, 5) for _ in range(n)]
    scores = [random.randint(1, 10**9) for _ in range(n)]
    print(*tags)
    print(*scores)
""",

"cc_405_E. Graph Cutting": """\
import random
random.seed(42)
# All vertices must have even degree for a valid partition to exist.
# Use a cycle (all degrees = 2, even).
n = 100000
edges = [(i, i % n + 1) for i in range(1, n+1)]  # single cycle
m = len(edges)
print(n, m)
for u, v in edges:
    print(u, v)
""",

"cc_716_D. Complete The Graph": """\
import random
random.seed(42)
n = 1000
s, t = 0, n - 1
L = 10**9
edges = []
# Path s->t with zero-weight edges (to be filled in)
for i in range(n - 1):
    edges.append((i, i+1, 0))
# Extra weighted edges
for _ in range(5000):
    u = random.randint(0, n-1)
    v = random.randint(0, n-1)
    if u != v:
        w = random.randint(1, 10**8)
        edges.append((u, v, w))
m = len(edges)
print(n, m, L, s, t)
for u, v, w in edges:
    print(u, v, w)
""",

"cc_920_E. Connected Components?": """\
import random
random.seed(42)
n = 100000
m = 100000
print(n, m)
pairs = set()
while len(pairs) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v:
        pairs.add((min(u,v), max(u,v)))
for u, v in pairs:
    print(u, v)
""",

"cc_949_C. Data Center Maintenance": """\
import random
random.seed(42)
n = 10000
k = 10000
q = 24
print(n, k, q)
print(*[random.randint(0, q-1) for _ in range(n)])
for _ in range(k):
    u = random.randint(1, n)
    v = random.randint(1, n)
    print(u, v)
""",

# ── EASY ──────────────────────────────────────────────────────────────────────

"cc_1060_A. Phone Numbers": """\
import random
random.seed(42)
n = 100
digits = ''.join([str(random.randint(0, 9)) for _ in range(n)])
print(n)
print(digits)
""",

"cc_1398_A. Bad Triangle": """\
import random
random.seed(42)
t = 3
print(t)
# Case 1: large sorted array
n = 200000
arr = sorted([random.randint(1, 10**9) for _ in range(n)])
print(n)
print(*arr)
# Case 2: all equal
print(3)
print(5, 5, 5)
# Case 3: impossible triangle
print(3)
print(1, 2, 10)
""",

"cc_168_A. Wizards and Demonstration": """\
import random
random.seed(42)
n = 100000
x = random.randint(1, n)
y = random.randint(1, 100)
print(n, x, y)
""",

"cc_615_A. Bulbs": """\
import random
random.seed(42)
n = 10
m = 100
print(n, m)
for i in range(n):
    cnt = random.randint(1, m)
    bs = random.sample(range(1, m+1), cnt)
    print(cnt, *bs)
""",

"cc_776_A. A Serial Killer": """\
import random, string
random.seed(42)
def rname():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 8)))
n = 50
a, b = rname(), rname()
print(a, b)
print(n)
for _ in range(n):
    victim = random.choice([a, b])
    replacement = rname()
    print(victim, replacement)
    if victim == a:
        a = replacement
    else:
        b = replacement
""",

"cc_868_A. Bark to Unlock": """\
import random, string
random.seed(42)
password = ''.join(random.choice(string.ascii_lowercase[:5]) for _ in range(2))
n = 100
words = set()
while len(words) < n:
    w = ''.join(random.choice(string.ascii_lowercase[:5]) for _ in range(random.randint(1, 5)))
    words.add(w)
print(password)
print(n)
for w in list(words)[:n]:
    print(w)
""",

"cc_915_A. Garden": """\
import random
random.seed(42)
n = 50
k = 100
print(n, k)
print(*[random.randint(1, k) for _ in range(n)])
""",

"cc_1043_A. Elections": """\
import random
random.seed(42)
n = 100
print(n)
print(*[random.randint(0, 100) for _ in range(n)])
""",

"cc_1236_A. Stones": """\
import random
random.seed(42)
t = 50
print(t)
seen = set()
count = 0
while count < t:
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    c = random.randint(0, 10)
    k = (a, b, c)
    if k not in seen:
        seen.add(k)
        print(a, b, c)
        count += 1
""",

"cc_1382_A. Common Subsequence": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    n = random.randint(50, 100)
    m = random.randint(50, 100)
    a = [random.randint(1, 10) for _ in range(n)]
    b = [random.randint(1, 10) for _ in range(m)]
    print(n, m)
    print(*a)
    print(*b)
""",

# ── MEDIUM ────────────────────────────────────────────────────────────────────

"cc_42_A. Guilty — to the kitchen!": """\
import random
random.seed(42)
n = 50
total = 10**6
print(n, total)
print(*[random.randint(1, 100) for _ in range(n)])
print(*[random.randint(1, 10**6) for _ in range(n)])
""",

"cc_940_A. Points on the line": """\
import random
random.seed(42)
n = 1000   # ref is O(n^2); keep small to avoid TLE
d = 5
print(n, d)
print(*[random.randint(1, 100) for _ in range(n)])
""",

"cc_1031_B. Curiosity Has No Limits": """\
import random
random.seed(42)
n = 5000
print(n)
print(*[random.randint(0, 3) for _ in range(n-1)])
print(*[random.randint(0, 3) for _ in range(n-1)])
""",

"cc_1076_B. Divisor Subtraction": """\
import random
random.seed(42)
print(random.randint(2, 10**10))
""",

"cc_1097_B. Petr and a Combination Lock": """\
import random
random.seed(42)
n = 15
print(n)
for _ in range(n):
    print(random.randint(1, 180))
""",

"cc_1305_B. Kuroni and Simple Strings": """\
import random
random.seed(42)
n = 1000   # ref uses recursion; depth ~n/2 so keep well under Python's 10000 limit
s = ''.join(random.choice('()') for _ in range(n))
print(s)
""",

"cc_181_B. Number of Triplets": """\
import random
random.seed(42)
n = 2000
pts = set()
while len(pts) < n:
    # ref uses cords[x+1000][y+1000] — coords must be in [-999, 999]
    pts.add((random.randint(-999, 999), random.randint(-999, 999)))
print(n)
for x, y in pts:
    print(x, y)
""",

"cc_278_B. New Problem": """\
import random, string
random.seed(42)
n = 30
print(n)
for _ in range(n):
    l = random.randint(1, 20)
    print(''.join(random.choice(string.ascii_lowercase) for _ in range(l)))
""",

"cc_443_B. Kolya and Tandem Repeat": """\
import random, string
random.seed(42)
s = ''.join(random.choice(string.ascii_lowercase) for _ in range(200))
k = 200
print(s)
print(k)
""",

"cc_489_B. BerSU Ball": """\
import random
random.seed(42)
n = 100
m = 100
print(n)
print(*[random.randint(1, 100) for _ in range(n)])
print(m)
print(*[random.randint(1, 100) for _ in range(m)])
""",

"cc_658_B. Bear and Displayed Friends": """\
import random
random.seed(42)
n = 100000
s = 5
q = 100000
print(n, s, q)
vals = random.sample(range(1, 10**9), n)
print(*vals)
for _ in range(q):
    # 1 = add friend idx, 2 = remove friend idx
    print(random.randint(1, 2), random.randint(1, n))
""",

"cc_747_C. Servers": """\
import random
random.seed(42)
n = 100   # ref is O(n*q) worst-case; keep small
q = 200
print(n, q)
t = 0
for _ in range(q):
    t += random.randint(1, 5)
    k = random.randint(1, min(n, 5))
    d = random.randint(1, 20)
    print(t, k, d)
""",

"cc_887_B. Cubes for Masha": """\
import random
random.seed(42)
n = 3000
print(n)
for _ in range(n):
    faces = [random.randint(0, 9) for _ in range(6)]
    print(*faces)
""",

"cc_1215_B. The Number of Products": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.choice([-1, 1]) * random.randint(1, 10**9) for _ in range(n)])
""",

"cc_177_D1. Encrypting Messages": """\
import random
random.seed(42)
n = 100000
m = random.randint(1, n)
mod = 10**9 + 7
print(n, m, mod)
print(*[random.randint(1, 10**6) for _ in range(n)])
print(*[random.randint(1, 10**6) for _ in range(m)])
""",

"cc_673_C. Bear and Colors": """\
import random
random.seed(42)
n = 3000
print(n)
print(*[random.randint(1, n) for _ in range(n)])
print(random.randint(1, n))
""",

"cc_740_B. Alyona and flowers": """\
import random
random.seed(42)
n = 1000   # ref is O(n*q); keep small to avoid TLE
q = 1000
print(n, q)
print(*[random.randint(-10**6, 10**6) for _ in range(n)])
for _ in range(q):
    l = random.randint(1, n)
    r = random.randint(l, n)
    print(l, r)
""",

"cc_1000_A. Codehorses T-shirts": """\
import random
random.seed(42)
sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXS', 'XXL', 'XXXS', 'XXXL']
n = 100
print(n)
for _ in range(n):
    print(random.choice(sizes))
for _ in range(n):
    print(random.choice(sizes))
""",

"cc_1139_C. Edgy Trees": """\
import random
random.seed(42)
n = 100000
k = random.randint(2, n)
print(n, k)
for i in range(2, n+1):
    p = random.randint(1, i-1)
    c = random.randint(0, 1)
    print(i, p, c)
""",

"cc_1238_B. Kill 'Em All": """\
import random
random.seed(42)
t = 2
print(t)
n = 200000
x = 10**6
positions = random.sample(range(-10**9, 10**9), n)
print(n, x)
print(*positions)
n2 = 100000
x2 = 5 * 10**5
positions2 = random.sample(range(-10**9, 10**9), n2)
print(n2, x2)
print(*positions2)
""",

# ── HARD ──────────────────────────────────────────────────────────────────────

"cc_20_B. Equation": """\
import random
random.seed(42)
A = random.randint(-10**5, 10**5)
B = random.randint(-10**5, 10**5)
C = random.randint(-10**5, 10**5)
print(A, B, C)
""",

"cc_593_C. Beautiful Function": """\
import random
random.seed(42)
n = 300   # ref uses recursion depth n; keep well under Python's limit
print(n)
for _ in range(n):
    x = random.randint(-10**9, 10**9)
    y = random.randint(-10**9, 10**9)
    print(x, y)  # ref reads 2 values per line: x y
""",

"cc_101_C. Vectors": """\
import random
random.seed(42)
x1 = random.randint(-10**9, 10**9)
y1 = random.randint(-10**9, 10**9)
x2 = random.randint(-10**9, 10**9)
y2 = random.randint(-10**9, 10**9)
cx = random.randint(-10**9, 10**9)
cy = random.randint(-10**9, 10**9)
print(x1, y1)
print(x2, y2)
print(cx, cy)
""",

"cc_578_C. Weakness and Poorness": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.randint(-10**6, 10**6) for _ in range(n)])
""",

"cc_621_D. Rat Kwesh and Cheese": """\
import random
random.seed(42)
x = round(random.uniform(0.1, 200.0), 1)
y = round(random.uniform(0.1, 200.0), 1)
z = round(random.uniform(0.1, 200.0), 1)
print(x, y, z)
""",

"cc_996_F. Game": """\
import random
random.seed(42)
n = 18
q = 1000
print(n, q)
print(*[random.randint(-10**9, 10**9) for _ in range(2**n)])
for _ in range(q):
    print(random.randint(0, n-1), random.randint(0, 1))
""",

"cc_1012_B. Chemical table": """\
import random
random.seed(42)
n = 500
m = 500
k = 200000
print(n, m, k)
pairs = set()
while len(pairs) < k:
    r = random.randint(1, n)
    c = random.randint(1, m)
    pairs.add((r, c))
for r, c in pairs:
    print(r, c)
""",

"cc_105_C. Item World": """\
import random, string
random.seed(42)
n = 21
# Guarantee at least one of each class so searchFor('armor',...) doesn't fail
fixed_classes = ['weapon', 'armor', 'orb']
extra_classes  = [random.choice(['weapon', 'armor', 'orb']) for _ in range(n - 3)]
all_classes    = fixed_classes + extra_classes
random.shuffle(all_classes)
print(n)
items = []
for cls in all_classes:
    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 6)))
    atk  = random.randint(1, 1000)
    dfn  = random.randint(1, 1000)
    res  = random.randint(1, 1000)
    lvl  = random.randint(1, 5)
    print(f'{name} {cls} {atk} {dfn} {res} {lvl}')
    items.append(name)
m = 20
print(m)
for _ in range(m):
    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 6)))
    cls  = random.choice(['gladiator', 'physician', 'sentry'])
    lvl  = random.randint(1, 100)
    item = random.choice(items)
    print(f'{name} {cls} {lvl} {item}')
""",

"cc_1149_B. Three Religions": """\
import random
random.seed(42)
# CF 1149B: one main string s of length n, then q operations on 3 religion strings
# Format: n q / s / [+ i c] or [- i]  (i in 1..3, c in a..z)
n = 10000
q = 1000
s = ''.join(random.choice('abcd') for _ in range(n))
print(n, q)
print(s)
lens = [0, 0, 0]
for _ in range(q):
    i = random.randint(1, 3)
    if lens[i - 1] > 0 and random.random() < 0.2:
        print(f'- {i}')
        lens[i - 1] -= 1
    else:
        c = random.choice('abcd')
        print(f'+ {i} {c}')
        lens[i - 1] += 1
""",

"cc_1354_C2. Not So Simple Polygon Embedding": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    # ref iterates O(n) per case; use small odd n to avoid TLE
    n = random.randrange(3, 1000, 2)
    print(n)
""",

}


def _run_script(script: str, stdin: bytes, timeout: int = 30) -> dict:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    with tempfile.NamedTemporaryFile(delete=False) as sf:
        sf.write(stdin)
        stdin_path = sf.name
    try:
        t0 = time.monotonic()
        with open(stdin_path, 'rb') as stdin_file:
            r = subprocess.run(
                ['python3', script_path],
                stdin=stdin_file,
                capture_output=True,
                timeout=timeout,
            )
        wall_ms = int((time.monotonic() - t0) * 1000)
        return {
            'ok': r.returncode == 0,
            'wall_ms': wall_ms,
            'stdout': r.stdout,
            'stderr': r.stderr.decode(errors='replace'),
        }
    except subprocess.TimeoutExpired:
        return {'ok': False, 'wall_ms': timeout * 1000, 'stdout': b'', 'stderr': 'TIMEOUT'}
    finally:
        os.unlink(script_path)
        try:
            os.unlink(stdin_path)
        except OSError:
            pass


def _run_generator(gen_script: str, timeout: int = 60) -> bytes | None:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(gen_script)
        path = f.name
    try:
        r = subprocess.run(
            ['python3', path],
            capture_output=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            print(f"    GENERATOR FAILED: {r.stderr.decode(errors='replace')[:200]}")
            return None
        return r.stdout
    except subprocess.TimeoutExpired:
        print(f"    GENERATOR TIMED OUT")
        return None
    finally:
        os.unlink(path)


def verify():
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    passed = 0
    failed = 0
    skipped = 0

    for problem in cache['problems']:
        tid = problem['task_id']
        if tid not in GENERATORS:
            print(f"[skip] {tid} — no generator")
            skipped += 1
            continue

        gen_script = GENERATORS[tid]
        ref = problem['ref_solution']

        print(f"[verify] {tid}")

        stdin_bytes = _run_generator(gen_script)
        if stdin_bytes is None:
            failed += 1
            continue

        result = _run_script(ref, stdin_bytes, timeout=30)
        if not result['ok']:
            print(f"  FAIL  wall={result['wall_ms']}ms stderr={result['stderr'][:200]}")
            failed += 1
        else:
            out = result['stdout']
            if not out.strip():
                print(f"  WARN  empty output wall={result['wall_ms']}ms")
            else:
                print(f"  OK    wall={result['wall_ms']}ms  output_len={len(out)}")
                problem['test_case_generator'] = gen_script
                passed += 1

    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"\n[verify] passed={passed}  failed={failed}  skipped={skipped}")
    print(f"[verify] {passed} generators embedded into {CACHE_PATH}")


if __name__ == '__main__':
    verify()
