#!/usr/bin/env python3
"""
generate_stress_tests.py

Writes hand-crafted test case generators for pool problems,
stores the generator script in the cache, then verifies each
generator by running it + the reference solution locally.

Usage:
    python3 online_rl/generate_stress_tests.py
"""

import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_PATH = "online_rl/cc_pool_cache.json"

GENERATORS = {

"cc_1101_A. Minimum Integer": """\
import random
random.seed(42)
q = 500
print(q)
for _ in range(q):
    d = random.randint(1, 2*10**9)
    l = random.randint(1, 2*10**9)
    r = random.randint(l, min(l + 10**9, 4*10**9))
    print(l, r, d)
""",

"cc_1189_D1. Add on a Tree": """\
import random
random.seed(42)
n = 100000
print(n)
for i in range(2, n + 1):
    parent = random.randint(1, i - 1)
    print(parent, i)
""",

"cc_1227_D1. Optimal Subsequences (Easy Version)": """\
import random
random.seed(42)
n, m = 100, 100
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
print(m)
for _ in range(m):
    k = random.randint(1, n)
    pos = random.randint(1, k)
    print(k, pos)
""",

"cc_1334_D. Minimum Euler Cycle": """\
import random
random.seed(42)
T = 100
print(T)
for _ in range(T):
    n = random.randint(2, 10)
    l = random.randint(1, n)
    r = random.randint(l, n)
    print(n, l, r)
""",

"cc_1374_E1. Reading Books (easy version)": """\
import random
random.seed(42)
n = 200000
k = random.randint(1, 10**18)
print(n, k)
for _ in range(n):
    t = random.randint(1, 10**9)
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    print(t, a, b)
""",

"cc_1421_B. Putting Bricks in the Wall": """\
import random
random.seed(42)
t = 10
print(t)
for _ in range(t):
    n = 500
    print(n)
    for i in range(n):
        row = []
        for j in range(n):
            if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                row.append('0')
            else:
                row.append(str(random.randint(0, 1)))
        print(''.join(row))
""",

"cc_143_A. Help Vasilisa the Wise 2": """\
import random
random.seed(42)
a = random.randint(1, 9)
b = random.randint(1, 9)
c = random.randint(1, 9)
d = random.randint(1, 9)
print(a + b, c + d)
print(a + c, b + d)
print(a + d, b + c)
""",

"cc_1513_C. Add One": """\
import random
random.seed(42)
t = 200000
print(t)
for _ in range(t):
    n = random.randint(1, 10**9)
    m = random.randint(1, 200000)
    print(n, m)
""",

"cc_260_B. Ancient Prophesy": """\
import random
random.seed(42)
parts = []
for _ in range(10000):
    r = random.random()
    if r < 0.3:
        dd = str(random.randint(1, 28)).zfill(2)
        mm = str(random.randint(1, 12)).zfill(2)
        yyyy = str(random.randint(2013, 2015))
        parts.append(f'{dd}-{mm}-{yyyy}')
    elif r < 0.6:
        parts.append(str(random.randint(10, 99)))
    else:
        parts.append('-')
print(''.join(parts))
""",

"cc_284_B. Cows and Poker Game": """\
import random
random.seed(42)
n = 200000
print(n)
n_allin = random.randint(1, n - 2)
n_in = random.randint(0, 2)
n_fold = n - n_allin - n_in
statuses = ['ALLIN'] * n_allin + ['IN'] * n_in + ['FOLDED'] * n_fold
random.shuffle(statuses)
print(' '.join(statuses))
""",

"cc_379_A. New Year Candles": """\
import random
random.seed(42)
a = 1000
b = 1000
print(a, b)
""",

"cc_39_H. Multiplication Table": """\
import random
random.seed(42)
k = 10
print(str(k).zfill(3))
""",

"cc_44_B. Cola": """\
import random
random.seed(42)
n = 10000
a = 5000
b = 5000
c = 5000
print(n, a, b, c)
""",

"cc_519_B. A and B and Compilation Errors": """\
import random
random.seed(42)
n = 100000
errors = [random.randint(1, 10**9) for _ in range(n)]
print(n)
print(*errors)
removed1 = random.randint(0, n-1)
errors2 = errors[:removed1] + errors[removed1+1:]
print(*errors2)
removed2 = random.randint(0, n-2)
errors3 = errors2[:removed2] + errors2[removed2+1:]
print(*errors3)
""",

"cc_545_C. Woodcutters": """\
import random
random.seed(42)
n = 100000
print(n)
x = 0
for _ in range(n):
    x += random.randint(1, 1000)
    h = random.randint(1, 10**6)
    print(x, h)
""",

"cc_634_C. Factory Repairs": """\
import random
random.seed(42)
n = 200000
a = 10000
b = random.randint(1, a-1)
k = random.randint(1, n-1)
q = 200000
print(n, k, a, b, q)
day = 1
for _ in range(q):
    if random.random() < 0.5:
        l = random.randint(day, day + 10)
        d = random.randint(1, 50)
        print(1, l, d)
        day = l
    else:
        n = random.randint(1, 50)
        print(2, n)
""",

"cc_663_A. Rebus": """\
import random
random.seed(42)
num_q = 100
ops = ['+'] + [random.choice(['+', '-']) for _ in range(num_q - 1)]
n = random.randint(1, 10**6)
parts = ['?']
for i in range(1, num_q):
    parts.append(ops[i])
    parts.append('?')
parts.append('=')
parts.append(str(n))
print(' '.join(parts))
""",

"cc_7_B. Memory Manager": """\
import random
random.seed(42)
m = 100
t = 100
print(t, m)
alloc_ids = []
next_id = 1
for _ in range(t):
    r = random.random()
    if r < 0.4:
        size = random.randint(1, m // 3)
        print('alloc', size)
        alloc_ids.append(next_id)
        next_id += 1
    elif r < 0.65 and alloc_ids:
        eid = random.choice(alloc_ids)
        print('erase', eid)
    elif r < 0.8:
        print('erase', random.randint(-5, next_id + 2))
    else:
        print('defragment')
""",

"cc_938_B. Run For Your Prize": """\
import random
random.seed(42)
n = 100000
print(n)
positions = random.sample(range(2, 10**6), n)
print(*positions)
""",

"cc_1107_D. Compression": """\
import random
random.seed(42)
n = 200  # divisible by 4; 200x200 binary matrix
print(n)
for _ in range(n):
    bits = ''.join(random.choice('01') for _ in range(n))
    val = int(bits, 2)
    print(format(val, f'0{n//4}X'))
""",

"cc_1136_D. Nastya Is Buying Lunch": """\
import random
random.seed(42)
n = 10000
m = 50000
print(n, m)
perm = list(range(1, n+1))
random.shuffle(perm)
print(*perm)
for _ in range(m):
    a = random.randint(1, n)
    b = random.randint(1, n)
    if a == b:
        b = b % n + 1
    print(a, b)
""",

"cc_1155_A. Reverse a Substring": """\
import random
random.seed(42)
import string
n = 100000
s = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
print(n)
print(s)
""",

"cc_1195_D2. Submarine in the Rybinsk Sea (hard edition)": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_151_C. Win or Freeze": """\
import random
random.seed(42)
# product of 3 distinct primes gives exactly 2 prime factors
primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
import random
a, b, c = 999999937, 999999929, 999999893
print(a * b * c)
""",

"cc_1547_C. Pair Programming": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    print()
    k = random.randint(0, 10)
    n = random.randint(1, 8)
    m = random.randint(1, 8)
    print(k, n, m)
    # a: monocarp actions (0=add line, positive=delete line i)
    lines_a = k
    a = []
    for _ in range(n):
        if lines_a > 0 and random.random() < 0.4:
            a.append(random.randint(1, lines_a))
        else:
            a.append(0)
            lines_a += 1
    print(*a)
    # b: polycarp actions
    lines_b = k
    b = []
    for _ in range(m):
        if lines_b > 0 and random.random() < 0.4:
            b.append(random.randint(1, lines_b))
        else:
            b.append(0)
            lines_b += 1
    print(*b)
""",

"cc_195_A. Let's Watch Football": """\
import random
random.seed(42)
a = 10**18
b = random.randint(1, a)
c = 10**9
print(a, b, c)
""",

"cc_219_A. k-String": """\
import random
random.seed(42)
import string
k = 1000
base = ''.join(random.choice(string.ascii_lowercase) for _ in range(1))
s = base * k
print(k)
print(s)
""",

"cc_242_C. King's Path": """\
import random
random.seed(42)
x0, y0 = random.randint(1, 20), random.randint(1, 20)
x1, y1 = random.randint(1, 20), random.randint(1, 20)
print(x0, y0, x1, y1)
n = 100000
print(n)
rows_used = set()
for _ in range(n):
    r = random.randint(min(x0,x1)-2, max(x0,x1)+2)
    a = random.randint(1, 50)
    b = random.randint(a, min(a+10, 50))
    print(r, a, b)
""",

"cc_290_D. Orange": """\
import random
random.seed(42)
import string
s = ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(5, 50)))
print(s)
print(random.randint(0, 26))
""",

"cc_316_B2. EKG": """\
import random
random.seed(42)
n = 1000
k = random.randint(1, n)
print(n, k)
# queue: n values, each 0..n (doctor references), but must be valid
vals = [0] * n
vals[random.randint(0, n-1)] = 1
print(*vals)
""",

"cc_361_C. Levko and Array Recovery": """\
import random
random.seed(42)
n = 100000
q = 100000
print(n, q)
for _ in range(q):
    op = random.randint(1, 2)
    l = random.randint(1, n)
    r = random.randint(l, n)
    if op == 1:
        d = random.randint(-100, 100)
        print(op, l, r, d)
    else:
        val = random.randint(1, 10**6)
        print(op, l, r, val)
""",

"cc_385_A. Bear and Raspberry": """\
import random
random.seed(42)
n = 30000
c = random.randint(1, 100)
print(n, c)
print(*[random.randint(1, 1000) for _ in range(n)])
""",

"cc_433_A. Kitahara Haruki's Gift": """\
import random
random.seed(42)
n = random.randint(2, 20)
print(n)
weights = [random.choice([100, 200]) for _ in range(n)]
print(*weights)
""",

"cc_478_A. Initial Bet": """\
import random
random.seed(42)
vals = [random.randint(1, 200) for _ in range(5)]
print(*vals)
""",

"cc_500_C. New Year Book Reading": """\
import random
random.seed(42)
n = 500
m = 1000
print(n, m)
print(*[random.randint(1, 100) for _ in range(n)])
order = [random.randint(1, n) for _ in range(m)]
print(*order)
""",

"cc_526_A. King of Thieves": """\
import random
random.seed(42)
n = random.randint(10, 100)
print(n)
s = ['*'] * n
# place one ) inside
pos = random.randint(1, n-2)
s[pos] = ')'
print(''.join(s))
""",

"cc_551_B. ZgukistringZ": """\
import random
random.seed(42)
import string
def rand_str(length):
    return ''.join(random.choice('abcde') for _ in range(length))
a = rand_str(100000)
b = rand_str(100000)
c = rand_str(100000)
print(a)
print(b)
print(c)
""",

"cc_5_A. Chat Server's Outgoing Traffic": """\
import random
random.seed(42)
import string
def rname():
    return ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(3, 8)))
def rmsg():
    return ' '.join(rname() for _ in range(random.randint(1, 5)))
members = []
ops = []
for _ in range(50):
    r = random.random()
    if r < 0.3 or not members:
        name = rname()
        members.append(name)
        ops.append('+' + name)
    elif r < 0.5:
        name = random.choice(members)
        members.remove(name)
        ops.append('-' + name)
    elif members:
        name = random.choice(members)
        ops.append(name + ':' + rmsg())
for op in ops:
    print(op)
""",

"cc_643_B. Bear and Two Paths": """\
import random
random.seed(42)
n = 1000
k = 2*n - 2
# a,b,c,d must be distinct and valid
nodes = random.sample(range(1, n+1), 4)
a, b, c, d = nodes
print(n, k)
print(a, b, c, d)
""",

"cc_670_B. Game of Robots": """\
import random
random.seed(42)
n = 100000
k = n*(n+1)//2
ids = random.sample(range(1, 10**9), n)
print(n, k)
print(*ids)
""",

"cc_691_D. Swaps in Permutation": """\
import random
random.seed(42)
n = 100000
m = 100000
perm = list(range(1, n+1))
random.shuffle(perm)
print(n, m)
print(*perm)
for _ in range(m):
    a = random.randint(1, n)
    b = random.randint(1, n)
    print(a, b)
""",

"cc_737_A. Road to Cinema": """\
import random
random.seed(42)
V = random.randint(2, 10)
S = random.randint(10, 500)
T = random.randint(1, 120)
k = random.randint(1, 10)
print(V, S, T, k)
for _ in range(V - 1):
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    print(a, b)
stations = sorted(random.sample(range(1, S), k))
print(*stations)
""",

"cc_805_A. Fake NP": """\
import random
random.seed(42)
l = random.randint(1, 10**9)
r = random.randint(l, min(l + 10**6, 10**9))
print(l, r)
""",

"cc_830_A. Office Keys": """\
import random
random.seed(42)
n = 1000
k = 2000
p = random.randint(1, 20000)
people = sorted(random.sample(range(1, 20000), n))
keys = sorted(random.sample(range(1, 20000), k))
print(n, k, p)
print(*people)
print(*keys)
""",

"cc_851_B. Arpa and an exam about geometry": """\
import random
random.seed(42)
coords = [random.randint(-10**9, 10**9) for _ in range(6)]
print(*coords)
""",

"cc_1037_B. Reach Median": """\
import random
random.seed(42)
n = 199999  # odd
s = random.randint(-10**9, 10**9)
print(n, s)
print(*[random.randint(-10**9, 10**9) for _ in range(n)])
""",

"cc_1081_D. Maximum Distance": """\
import random
random.seed(42)
n = 100000
m = 100000
k = random.randint(1, n)
specials = random.sample(range(1, n+1), k)
print(n, m, k)
print(*specials)
edges = set()
# ensure connected: chain
for i in range(1, n):
    edges.add((i, i+1, random.randint(1, 10**9)))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v:
        edges.add((min(u,v), max(u,v), random.randint(1, 10**9)))
for u, v, w in list(edges)[:m]:
    print(u, v, w)
""",

"cc_1129_A2. Toy Train": """\
import random
random.seed(42)
n = 5000
m = 100000
print(n, m)
for _ in range(m):
    s = random.randint(1, n)
    d = random.randint(1, n)
    print(s, d)
""",

"cc_1227_A. Math Problem": """\
import random
random.seed(42)
t = 5
print(t)
for _ in range(t):
    n = 300000
    print(n)
    for _ in range(n):
        l = random.randint(0, 10**9)
        r = random.randint(l, min(l + 10**9, 2 * 10**9))
        print(l, r)
""",

"cc_124_D. Squares": """\
import random
random.seed(42)
a = random.randint(1, 10**9)
b = random.randint(1, 10**9)
x1 = random.randint(-10**9, 10**9)
y1 = random.randint(-10**9, 10**9)
x2 = random.randint(-10**9, 10**9)
y2 = random.randint(-10**9, 10**9)
print(a, b, x1, y1, x2, y2)
""",

"cc_1269_B. Modulo Equality": """\
import random
random.seed(42)
n = 200000
m = 10**9
a = [random.randint(0, m-1) for _ in range(n)]
b = [random.randint(0, m-1) for _ in range(n)]
print(n, m)
print(*a)
print(*b)
""",

"cc_1291_B. Array Sharpening": """\
import random
random.seed(42)
t = 500
print(t)
for _ in range(t):
    n = random.randint(1, 1000)
    print(n)
    print(*[random.randint(0, 10**6) for _ in range(n)])
""",

"cc_1311_C. Perform the Combo": """\
import random
random.seed(42)
import string
t = 4
print(t)
for _ in range(t):
    n = 200000
    m = random.randint(1, 10)
    s = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
    print(n, m)
    print(s)
    fails = sorted(random.sample(range(n), min(m, n)))
    print(*fails)
""",

"cc_1334_A. Level Statistics": """\
import random
random.seed(42)
T = 500
print(T)
for _ in range(T):
    n = 100
    print(n)
    plays, clears = 0, 0
    for _ in range(n):
        dp = random.randint(0, 5)
        dc = random.randint(0, dp)
        plays += dp
        clears += dc
        print(plays, clears)
""",

"cc_1397_C. Multiples of Length": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(-10**9, 10**9) for _ in range(n)])
""",

"cc_1420_D. Rescue Nibel!": """\
import random
random.seed(42)
n = random.randint(5, 50)
k = random.randint(1, n)
print(n, k)
for _ in range(k):
    l = random.randint(1, 10**5)
    r = random.randint(l, l + 10**5)
    print(l, r)
""",

"cc_1466_C. Canine poetry": """\
import random
random.seed(42)
import string
t = 100000
print(t)
for _ in range(t):
    n = random.randint(1, 5)
    s = ''.join(random.choice('abcde') for _ in range(n))
    print(s)
""",

"cc_1490_D. Permutation Transformation": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = 100
    print(n)
    perm = list(range(1, n+1))
    random.shuffle(perm)
    print(*perm)
""",

"cc_235_A. LCM Challenge": """\
import random
random.seed(42)
print(10**6)
""",

"cc_378_C. Maze": """\
import random
random.seed(42)
n, m = 10, 10
k = random.randint(1, 5)
print(n, m, k)
for i in range(n):
    row = ''.join(random.choice('..#') for _ in range(m))
    print(row)
""",

"cc_425_A. Sereja and Swaps": """\
import random
random.seed(42)
n = 200
k = 10
print(n, k)
print(*[random.randint(-1000, 1000) for _ in range(n)])
""",

"cc_494_A. Treasure": """\
import random
random.seed(42)
n = random.randint(20, 80)
# build balanced with some #s
parts = []
depth = 0
for i in range(n):
    if depth == 0 or (random.random() < 0.5 and i < n - depth):
        parts.append('(')
        depth += 1
    else:
        parts.append(')')
        depth -= 1
while depth > 0:
    parts.append(')')
    depth -= 1
# replace some ) with #
s = list(''.join(parts))
for i in range(len(s)):
    if s[i] == ')' and random.random() < 0.2:
        s[i] = '#'
print(''.join(s))
""",

"cc_687_A. NP-Hard Problem": """\
import random
random.seed(42)
n = 100000
m = 100000
print(n, m)
edges = set()
for i in range(1, n):
    edges.add((i, i+1))
while len(edges) < m:
    u = random.randint(1, n)
    v = random.randint(1, n)
    if u != v:
        edges.add((min(u,v), max(u,v)))
for u, v in edges:
    print(u, v)
""",

"cc_730_G. Car Repair Shop": """\
import random
random.seed(42)
n = 200
print(n)
for _ in range(n):
    s = random.randint(1, 10**9)
    d = random.randint(1, 1000)
    print(s, d)
""",

"cc_754_B. Ilya and tic-tac-toe game": """\
import random
random.seed(42)
grid = [['.' for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        grid[i][j] = random.choice(['x', 'o', '.'])
for row in grid:
    print(''.join(row))
""",

"cc_774_K. Stepan and Vowels": """\
import random
random.seed(42)
import string
n = 100000
vowels = 'aeiouy'
s = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
print(n)
print(s)
""",

"cc_846_B. Math Show": """\
import random
random.seed(42)
n = 45
k = 45
m = 2*10**9
print(n, k, m)
print(*sorted([random.randint(1, 10**6) for _ in range(k)]))
""",

"cc_990_B. Micro-World": """\
import random
random.seed(42)
n = 200000
k = 10**6
print(n, k)
print(*[random.randint(1, 10**6) for _ in range(n)])
""",

"cc_1165_A. Remainder": """\
import random
random.seed(42)
x = 100
y = 50
n = 100000
s = ''.join(random.choice('01') for _ in range(n))
print(n, x, y)
print(s)
""",

"cc_1264_A. Beautiful Regional Contest": """\
import random
random.seed(42)
t = 50
print(t)
for _ in range(t):
    n = random.randint(3, 5000)
    print(n)
    scores = sorted([random.randint(0, 20) for _ in range(n)], reverse=True)
    print(*scores)
""",

"cc_1285_C. Fadi and LCM": """\
import random
random.seed(42)
# product of two large primes
primes = [999999937, 999999929, 999999893, 999999883, 1000003, 1000033]
a, b = 999999937, 999999893
print(a * b)
""",

"cc_1391_C. Cyclic Permutations ": """\
import random
random.seed(42)
print(10**6)
""",

"cc_161_B. Discounts": """\
import random
random.seed(42)
n = random.randint(5, 30)
k = random.randint(1, n-1)
print(n, k)
for _ in range(n):
    c = random.randint(1, 10**9)
    t = random.randint(1, 2)
    print(c, t)
""",

"cc_252_B. Unsorting Array": """\
import random
random.seed(42)
n = 100000
vals = random.sample(range(-(10**9), 10**9), n)
print(n)
print(*vals)
""",

"cc_348_A. Mafia": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(1, 1000) for _ in range(n)])
""",

"cc_371_B. Fox Dividing Cheese": """\
import random
random.seed(42)
a = random.randint(1, 10**9)
b = random.randint(1, 10**9)
print(a, b)
""",

"cc_488_C. Fight the Monster": """\
import random
random.seed(42)
hp_y = random.randint(10, 200)
at_y = random.randint(1, 100)
df_y = random.randint(1, 100)
print(hp_y, at_y, df_y)
hp_m = random.randint(10, 200)
at_m = random.randint(1, 100)
df_m = random.randint(1, 100)
print(hp_m, at_m, df_m)
cst_hp = random.randint(1, 10)
cst_at = random.randint(1, 10)
cst_df = random.randint(1, 10)
print(cst_hp, cst_at, cst_df)
""",

"cc_560_D. Equivalent Strings": """\
import random
random.seed(42)
import string
# length must be power-of-2 for the recursion to work nicely
n = 2 ** 17
a = ''.join(random.choice('abc') for _ in range(n))
b = ''.join(random.choice('abc') for _ in range(n))
print(a)
print(b)
""",

"cc_609_B. The Best Gift": """\
import random
random.seed(42)
m = 10
n = 200000
print(n, m)
genres = [random.randint(1, m) for _ in range(n)]
print(*genres)
""",

"cc_814_C. An impassioned circulation of affection": """\
import random
random.seed(42)
import string
n = 1500
s = ''.join(random.choice(string.ascii_lowercase[:6]) for _ in range(n))
q = 200000
print(n)
print(s)
print(q)
for _ in range(q):
    k = random.randint(0, n // 2)
    c = random.choice(string.ascii_lowercase[:6])
    print(k, c)
""",

"cc_985_A. Chess Placing": """\
import random
random.seed(42)
n = 100
print(n)
positions = sorted(random.sample(range(1, n+1), n // 2))
print(*positions)
""",

"cc_1003_D. Coins and Queries": """\
import random
random.seed(42)
n = 200000
q = 200000
print(n, q)
coins = [2 ** random.randint(0, 30) for _ in range(n)]
print(*coins)
total = sum(coins)
for _ in range(q):
    print(random.randint(1, total))
""",

"cc_1140_B. Good String": """\
import random
random.seed(42)
t = 15
print(t)
for _ in range(t):
    n = random.randint(1, 20)
    s = ''.join(random.choice('<>') for _ in range(n))
    print(n)
    print(s)
""",

"cc_1216_D. Swords": """\
import random
random.seed(42)
n = 200000
print(n)
vals = sorted(random.sample(range(1, 10**9), n))
print(*vals)
""",

"cc_1281_B. Azamon Web Services": """\
import random
random.seed(42)
import string
t = 1500
print(t)
for _ in range(t):
    def rword():
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(3, 10)))
    print(rword(), rword())
""",

"cc_1301_B. Motarack's Birthday": """\
import random
random.seed(42)
t = 10
print(t)
for _ in range(t):
    n = 5000
    print(n)
    arr = [random.randint(0, 50) for _ in range(n)]
    # randomly set some to -1
    for i in range(n):
        if random.random() < 0.3:
            arr[i] = -1
    print(*arr)
""",

"cc_1344_A. Hilbert's Hotel": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    n = 100
    print(n)
    print(*[random.randint(-200, 200) for _ in range(n)])
""",

"cc_1366_A. Shovels and Swords": """\
import random
random.seed(42)
t = 1000
print(t)
for _ in range(t):
    a = 10**9
    b = 10**9
    print(a, b)
""",

"cc_1451_D. Circle Game": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    d = 10**9
    k = random.randint(1, 10**9)
    print(d, k)
""",

"cc_1475_D. Cleaning the Phone": """\
import random
random.seed(42)
t = 20
print(t)
for _ in range(t):
    n = random.randint(3, 5000)
    m = random.randint(1, 50)
    print(n, m)
    a = [random.randint(1, 50) for _ in range(n)]
    b = [random.randint(1, 2) for _ in range(n)]
    print(*a)
    print(*b)
""",

"cc_1525_D. Armchairs": """\
import random
random.seed(42)
n = 5000
arr = [0] * n
# put at most n//2 people
num_people = random.randint(1, n // 2)
people_pos = random.sample(range(n), num_people)
for p in people_pos:
    arr[p] = 1
print(n)
print(*arr)
""",

"cc_178_A1. Educational Game": """\
import random
random.seed(42)
n = 300
print(n)
print(*[random.randint(0, 1000) for _ in range(n)])
""",

"cc_272_B. Dima and Sequence": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_343_B. Alternating Current": """\
import random
random.seed(42)
n = 100000
s = ''.join(random.choice('+-') for _ in range(n))
print(s)
""",

"cc_38_C. Blinds": """\
import random
random.seed(42)
m = random.randint(10, 100)
l = random.randint(1, 100)
print(m, l)
print(*[random.randint(1, 100) for _ in range(m)])
""",

"cc_40_B. Repaintings": """\
import random
random.seed(42)
n = random.randint(3, 10**6)
m = random.randint(3, 10**6)
print(n, m)
x = random.randint(1, min(n, m) // 2 + 1)
print(x)
""",

"cc_438_A. The Child and Toy": """\
import random
random.seed(42)
n = 1000
m = 2000
print(n, m)
print(*[random.randint(1, 1000) for _ in range(n)])
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

"cc_483_B. Friends and Presents": """\
import random
random.seed(42)
cnt1 = random.randint(1, 10**6)
cnt2 = random.randint(1, 10**6)
x = random.randint(2, 1000)
y = random.randint(2, 1000)
while x == y:
    y = random.randint(2, 1000)
print(cnt1, cnt2, x, y)
""",

"cc_507_B. Amr and Pins": """\
import random
random.seed(42)
r = random.randint(1, 10**5)
x = random.randint(-10**5, 10**5)
y = random.randint(-10**5, 10**5)
xp = random.randint(-10**5, 10**5)
yp = random.randint(-10**5, 10**5)
print(r, x, y, xp, yp)
""",

"cc_556_C. Case of Matryoshkas": """\
import random
random.seed(42)
n = random.randint(3, 10)
m = random.randint(2, 5)
print(n, m)
for _ in range(m):
    length = random.randint(1, n)
    print(length, *random.sample(range(1, n+1), length))
""",

"cc_582_A. GCD Table": """\
import random
random.seed(42)
n = random.randint(2, 4)
print(n)
a = [random.randint(1, 100) for _ in range(n)]
from math import gcd
row = []
for i in range(n):
    for j in range(n):
        row.append(gcd(a[i], a[j]))
print(*row)
""",

"cc_604_A. Uncowed Forces": """\
import random
random.seed(42)
t = [random.randint(0, 120) for _ in range(5)]
w = [random.randint(0, 10) for _ in range(5)]
q = random.randint(0, 100)
z = random.randint(0, 50)
print(*t)
print(*w)
print(q, z)
""",

"cc_626_D. Jerry's Protest": """\
import random
random.seed(42)
n = 2000
print(n)
vals = random.sample(range(1, 5001), n)
print(*vals)
""",

"cc_765_D. Artsem and Saunders": """\
import random
random.seed(42)
n = 100000
print(n)
# f must satisfy f(f(x)) = f(x)
# choose some fixed points
fixed = random.sample(range(1, n+1), random.randint(1, n//2 + 1))
f = {}
for x in range(1, n+1):
    f[x] = random.choice(fixed)
print(*[f[x] for x in range(1, n+1)])
""",

"cc_789_A. Anastasia and pebbles": """\
import random
random.seed(42)
n = 100000
k = 10**9
print(n, k)
print(*[random.randint(1, 1000) for _ in range(n)])
""",

"cc_80_C. Heroes": """\
import random
random.seed(42)
names = ['Anka', 'Chapay', 'Cleo', 'Dracul', 'Troll', 'Snowy', 'Hexadecimal']
pairs = []
for i, a in enumerate(names):
    for b in names:
        if a != b:
            pairs.append(f'{a} likes {b}')
random.shuffle(pairs)
n = random.randint(5, len(pairs))
print(n)
for p in pairs[:n]:
    print(p)
""",

"cc_87_B. Vasya and Types": """\
import random
random.seed(42)
import string
def rname():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 8)))
n = random.randint(5, 20)
print(n)
typedefs = {'void': 'void'}
names = list(typedefs.keys())
for _ in range(n):
    if random.random() < 0.6 and names:
        src = random.choice(names)
        prefix = '&' * random.randint(0, 3)
        stars = '*' * random.randint(0, 3)
        if random.random() < 0.5:
            newname = rname()
            print(f'typedef {prefix}{src}{stars} {newname}')
            typedefs[newname] = newname
            names.append(newname)
        else:
            print(f'typeof {prefix}{src}{stars}')
    else:
        src = random.choice(list(typedefs.keys()))
        print(f'typeof {src}')
""",

"cc_1036_B. Diagonal Walking v.2": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    n = random.randint(-10**9, 10**9)
    m = random.randint(-10**9, 10**9)
    k = random.randint(abs(n) + abs(m), abs(n) + abs(m) + 10)
    print(n, m, k)
""",

"cc_1059_A. Cashier": """\
import random
random.seed(42)
n = random.randint(0, 30)
L = random.randint(100, 1000)
a = random.randint(1, 50)
print(n, L, a)
t = 0
for _ in range(n):
    t += random.randint(1, 20)
    l = random.randint(1, 20)
    print(t, l)
    t += l
""",

"cc_1080_C. Masha and two friends": """\
import random
random.seed(42)
t = 1000
print(t)
for _ in range(t):
    n = random.randint(2, 10**9)
    m = random.randint(2, 10**9)
    print(n, m)
    a1, b1 = random.randint(1, n), random.randint(1, m)
    c1, d1 = random.randint(a1, n), random.randint(b1, m)
    a2, b2 = random.randint(1, n), random.randint(1, m)
    c2, d2 = random.randint(a2, n), random.randint(b2, m)
    print(a1, b1, c1, d1)
    print(a2, b2, c2, d2)
""",

"cc_10_B. Cinema Cashier": """\
import random
random.seed(42)
k = 99
n = 1000
print(n, k)
print(*[random.randint(1, k) for _ in range(n)])
""",

"cc_1121_A. Technogoblet of Fire": """\
import random
random.seed(42)
m = 100
n = 100
k = random.randint(1, n)
print(n, m, k)
powers = [random.randint(1, 200) for _ in range(n)]
schools = [random.randint(1, m) for _ in range(n)]
favorites = random.sample(range(1, n+1), k)
print(*powers)
print(*schools)
print(*favorites)
""",

"cc_1148_C. Crazy Diamond": """\
import random
random.seed(42)
n = 300000
print(n)
perm = list(range(1, n+1))
random.shuffle(perm)
print(*perm)
""",

"cc_1225_B1. TV Subscriptions (Easy Version)": """\
import random
random.seed(42)
x = 1000
print(x)
for _ in range(x):
    k = random.randint(2, 100)
    n = 100
    d = random.randint(1, n)
    print(n, k, d)
    print(*[random.randint(1, k) for _ in range(n)])
""",

"cc_1249_D1. Too Many Segments (easy version)": """\
import random
random.seed(42)
n = 200
k = 5
print(n, k)
for _ in range(n):
    l = random.randint(0, 100)
    r = random.randint(l, l + 20)
    print(l, r)
""",

"cc_1267_L. Lexicography": """\
import random
random.seed(42)
import string
n = 1000
l = 1000
k = random.randint(1, n)
print(n, l, k)
print(''.join(random.choice(string.ascii_lowercase) for _ in range(n * l)))
""",

"cc_1290_A. Mind Control": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = random.randint(2, 1000)
    pos = random.randint(1, n)
    control = random.randint(0, pos - 1)
    print(n, pos, control)
    print(*[random.randint(-100, 100) for _ in range(n)])
""",

"cc_1353_C. Board Moves": """\
import random
random.seed(42)
t = 200
print(t)
for _ in range(t):
    n = random.randint(0, 249999) * 2 + 1  # odd
    print(n)
""",

"cc_1396_A. Multiples of Length": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(-10**9, 10**9) for _ in range(n)])
""",

"cc_141_C. Queue": """\
import random
random.seed(42)
import string
def rname():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(2, 8)))
n = 3000
print(n)
names = [rname() for _ in range(n)]
for i, name in enumerate(names):
    count = random.randint(0, i)
    print(name, count)
""",

"cc_1463_B. Find The Array": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = 50
    print(n)
    print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_148_B. Escape": """\
import random
random.seed(42)
vp = random.randint(1, 100)
vd = random.randint(1, 100)
t = random.randint(1, 100)
f = random.randint(1, 20)
c = random.randint(100, 2000)
print(vp)
print(vd)
print(t)
print(f)
print(c)
""",

"cc_1539_D. PriceFixed": """\
import random
random.seed(42)
n = 100000
print(n)
for _ in range(n):
    a = random.randint(1, 10**9)
    b = random.randint(1, 10**9)
    print(a, b)
""",

"cc_166_A. Rank List": """\
import random
random.seed(42)
n = random.randint(5, 50)
k = random.randint(1, n)
print(n, k)
for _ in range(n):
    p = random.randint(0, 50)
    t = random.randint(0, 50)
    print(p, t)
""",

"cc_233_C. Cycles": """\
import random
random.seed(42)
k = 100000
print(k)
""",

"cc_259_B. Little Elephant and Magic Square": """\
import random
random.seed(42)
# generate valid magic square then zero out one cell
center = random.randint(5, 20)
sq = [
    [center-1, center+2, center-1],  # won't always be magic, use known pattern
    [center+2, center, center-2],
    [center-1, center-2, center+3],
]
# Use standard magic square around center
c = center
magic = [
    [c+3, c-4, c+1],
    [c-2, c,   c+2],
    [c+1, c+4, c-5],
]
# zero out one cell
r, col = random.randint(0, 2), random.randint(0, 2)
magic[r][col] = 0
for row in magic:
    print(*row)
""",

"cc_282_B. Painting Eggs": """\
import random
random.seed(42)
n = 100000
print(n)
for _ in range(n):
    g = random.randint(-10**9, 10**9)
    a = random.randint(-10**9, 10**9)
    print(g, a)
""",

"cc_305_A. Strange Addition": """\
import random
random.seed(42)
n = random.randint(10, 100)
vals = random.sample(range(0, 101), min(n, 101))
print(n)
print(*vals)
""",

"cc_331_A2. Oh Sweet Beaverette": """\
import random
random.seed(42)
n = 100000
# generate array with some repeated values for interval structure
vals = [random.randint(-10**9, 10**9) for _ in range(n)]
# duplicate some values
for i in range(0, n-1, 3):
    vals[i+1] = vals[i]
print(n)
print(*vals)
""",

"cc_377_A. Maze": """\
import random
random.seed(42)
n, m = 10, 10
k = random.randint(1, 5)
print(n, m, k)
# start with all empty
grid = [['.' for _ in range(m)] for _ in range(n)]
# add some walls but keep connectivity
for i in range(n):
    for j in range(m):
        if random.random() < 0.2:
            grid[i][j] = '#'
for row in grid:
    print(''.join(row))
""",

"cc_448_B. Suffix Structures": """\
import random
random.seed(42)
import string
n = random.randint(20, 80)
s = ''.join(random.choice(string.ascii_lowercase[:8]) for _ in range(n))
t_len = random.randint(5, n)
t = ''.join(random.choice(string.ascii_lowercase[:8]) for _ in range(t_len))
print(s)
print(t)
""",

"cc_46_C. Hamsters and Tigers": """\
import random
random.seed(42)
n = 1000
s = ''.join(random.choice('HT') for _ in range(n))
print(n)
print(s)
""",

"cc_492_D. Vanya and Computer Game": """\
import random
random.seed(42)
n = 100000
x = 10**6
y = 10**6
print(n, x, y)
for _ in range(n):
    print(random.randint(1, 50))
""",

"cc_591_C. Median Smoothing": """\
import random
random.seed(42)
n = 500000
print(n)
print(*[random.randint(0, 1) for _ in range(n)])
""",

"cc_633_A. Ebony and Ivory": """\
import random
random.seed(42)
a = random.randint(1, 1000)
b = random.randint(1, 1000)
c = random.randint(1, 10000)
print(a, b, c)
""",

"cc_774_D. Lie or Truth": """\
import random
random.seed(42)
n = 100000
l = random.randint(1, n-1)
r = random.randint(l, n)
print(n, l, r)
a = [random.randint(1, 10) for _ in range(n)]
# b = same as a but segment [l-1..r-1] shuffled
b = a[:]
segment = b[l-1:r]
random.shuffle(segment)
b[l-1:r] = segment
print(*a)
print(*b)
""",

"cc_845_B. Luba And The Ticket": """\
import random
random.seed(42)
digits = [random.randint(0, 9) for _ in range(6)]
print(''.join(map(str, digits)))
""",

"cc_936_A. Save Energy!": """\
import random
random.seed(42)
k = random.randint(1, 10**9)
d = random.randint(1, k)
t = random.randint(1, 10**18)
print(k, d, t)
""",

"cc_1016_C. Vasya And The Mushrooms": """\
import random
random.seed(42)
n = 300000
print(n)
print(*[random.randint(0, 100) for _ in range(n)])
print(*[random.randint(0, 100) for _ in range(n)])
""",

"cc_1062_D. Fun with Integers": """\
import random
random.seed(42)
print(10**6)
""",

"cc_1084_C. The Fair Nut and String": """\
import random
random.seed(42)
n = 100000
s = ''.join(random.choice('ab') for _ in range(n))
print(s)
""",

"cc_1191_C. Tokitsukaze and Discard Items": """\
import random
random.seed(42)
n = 10**18
m = 200000
k = 10**9
print(n, m, k)
items = sorted(random.sample(range(1, n+1), m))
print(*items)
""",

"cc_1271_C. Shawarma Tent": """\
import random
random.seed(42)
n = 200000
x = random.randint(1, 10**6)
y = random.randint(1, 10**6)
print(n, x, y)
for _ in range(n):
    a = random.randint(x - 10, x + 10)
    b = random.randint(y - 10, y + 10)
    print(a, b)
""",

"cc_1294_D. MEX maximizing": """\
import random
random.seed(42)
q = 400000
x = 400000
print(q, x)
for _ in range(q):
    print(random.randint(0, 2 * x))
""",

"cc_1315_D. Recommendations": """\
import random
random.seed(42)
n = 200000
print(n)
a = [random.randint(1, 20) for _ in range(n)]
b = [random.randint(1, 10**9) for _ in range(n)]
print(*a)
print(*b)
""",

"cc_1468_J. Road Reform": """\
import random
random.seed(42)
t = 4
print(t)
for _ in range(t):
    n = random.randint(3, 8)
    m = random.randint(n-1, min(n*(n-1)//2, 12))
    k = random.randint(1, 10**9)
    print(n, m, k)
    edges = set()
    for i in range(1, n):
        edges.add((i, i+1, random.randint(1, 10**9)))
    while len(edges) < m:
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u != v:
            edges.add((min(u,v), max(u,v), random.randint(1, 10**9)))
    for u, v, s in list(edges)[:m]:
        print(u, v, s)
""",

"cc_191_A. Dynasty Puzzles": """\
import random
random.seed(42)
import string
def rword(min_l=2, max_l=8):
    return ''.join(random.choice(string.ascii_lowercase[:10]) for _ in range(random.randint(min_l, max_l)))
n = 500000
print(n)
for _ in range(n):
    print(rword())
""",

"cc_239_A. Two Bags of Potatoes": """\
import random
random.seed(42)
k = random.randint(2, 1000)
n = random.randint(k * 2, 10**9)
y = random.randint(1, n - k)
print(y, k, n)
""",

"cc_263_D. Cycle in Graph": """\
import random
random.seed(42)
n = 100000
k = 4
# build graph where each node has degree >= k
# use random k-regular-ish graph
m_target = n * k // 2 + random.randint(0, 20)
edges = set()
adj = [set() for _ in range(n+1)]
# ensure min degree k: connect each to k random others
for u in range(1, n+1):
    while len(adj[u]) < k:
        v = random.randint(1, n)
        if v != u:
            adj[u].add(v)
            adj[v].add(u)
            edges.add((min(u,v), max(u,v)))
m = len(edges)
print(n, m, k)
for u, v in edges:
    print(u, v)
""",

"cc_358_B. Dima and Text Messages": """\
import random
random.seed(42)
import string
def rword():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 6)))
n = 100000
words = [rword() for _ in range(n)]
print(n)
for w in words:
    print(w)
message = '<3' + '<3'.join(words) + '<3'
# randomly drop some chars
result = []
for c in message:
    if random.random() > 0.05:
        result.append(c)
print(''.join(result))
""",

# ── 50 new stress problems ────────────────────────────────────────────────────

"cc_710_E. Generate a String": """\
import random
random.seed(42)
n = 10000000
x = random.randint(1, 10**9)
y = random.randint(1, 10**9)
print(n, x, y)
""",

"cc_204_D. Little Elephant and Retro Strings": """\
import random
random.seed(42)
n = 500000
k = random.randint(1, n // 3)
s = ''.join(random.choice('BWX') for _ in range(n))
print(n, k)
print(s)
""",

"cc_134_B. Pairs of Numbers": """\
print(1000000)
""",

"cc_p00165 Lottery": """\
import random
random.seed(42)
for _ in range(10):
    n = 50
    print(n)
    for _ in range(n):
        p = random.randint(2, 999983)
        m = random.randint(-100, 100)
        print(p, m)
print(0)
""",

"cc_1263_E. Editor": """\
import random
random.seed(42)
n = 500000
ops = []
for _ in range(n):
    ops.append(random.choice(['(', ')', 'L', 'R', 'a', 'b', 'c', 'd']))
print(n)
print(''.join(ops))
""",

"cc_1292_D. Chaotic V.": """\
import random
random.seed(42)
n = 500000
print(n)
print(*[random.randint(1, 5000) for _ in range(n)])
""",

"cc_209_A. Multicolored Marbles": """\
print(1000000)
""",

"cc_317_D. Game with Powers": """\
print(1000000000)
""",

"cc_587_A. Duff and Weight Lifting": """\
import random
random.seed(42)
n = 1000000
print(n)
print(*[random.randint(0, 999999) for _ in range(n)])
""",

"cc_588_C. Duff and Weight Lifting": """\
import random
random.seed(42)
n = 1000000
print(n)
print(*[random.randint(0, 999999) for _ in range(n)])
""",

"cc_1106_F. Lunar New Year and a Recursive Sequence": """\
import random
random.seed(42)
n = 10**9
k = 100
bs = [random.randint(0, 10**9) for _ in range(k - 1)]
fn = random.randint(1, 998244352)
print(n, k)
print(*bs)
print(fn)
""",

"cc_39_E. What Has Dirichlet Got to Do with That?": """\
import random
random.seed(42)
a = 10000
b = 30
n = random.randint(10**8, 10**9)
print(a, b, n)
""",

"cc_1062_B. Math": """\
print(1000000)
""",

"cc_1208_A. XORinacci": """\
import random
random.seed(42)
T = 1000
print(T)
for _ in range(T):
    a = random.randint(0, 10**9)
    b = random.randint(0, 10**9)
    n = random.randint(0, 10**9)
    print(a, b, n)
""",

"cc_p00865 Expected Allowance": """\
import random
random.seed(42)
for _ in range(30):
    n = random.randint(1, 13)
    m = random.randint(1, 2008)
    k = random.randint(-n * m, n * m)
    print(n, m, k)
print(0, 0, 0)
""",

"cc_1422_D. Returning Home": """\
import random
random.seed(42)
n = 10**9
m = 100000
sx = random.randint(1, n)
sy = random.randint(1, n)
fx = random.randint(1, n)
fy = random.randint(1, n)
print(n, m)
print(sx, sy, fx, fy)
for _ in range(m):
    print(random.randint(1, n), random.randint(1, n))
""",

"cc_471_C. MUH and House of Cards": """\
print(1000000000)
""",

"cc_271_C. Secret": """\
import random
random.seed(42)
n = 1000000
k = random.randint(1, n // 3)
print(n, k)
""",

"cc_964_A. Splits": """\
print(1000000000)
""",

"cc_p01295 Champernowne Constant": """\
import random
random.seed(42)
for _ in range(50):
    N = random.randint(1, 10**9)
    K = random.randint(1, 100)
    print(N, K)
print(0, 0)
""",

"cc_401_C. Team": """\
import random
random.seed(42)
n0 = 333333
n1 = 333334
print(n0, n1)
""",

"cc_p03212 AtCoder Beginner Contest 114 - 755": """\
print(999999999)
""",

"cc_1244_G. Running in Pairs": """\
import random
random.seed(42)
n = 1000000
t_min = n * (n + 1) // 2
t = t_min + random.randint(0, 10**12)
print(n, t)
""",

"cc_1476_A. K-divisible Sum": """\
import random
random.seed(42)
T = 1000
print(T)
for _ in range(T):
    n = random.randint(1, 10**9)
    k = random.randint(1, 10**9)
    print(n, k)
""",

"cc_1366_B. Shuffle": """\
import random
random.seed(42)
T = 100
print(T)
for _ in range(T):
    n = 10**9
    x = random.randint(1, n)
    m = 100
    print(n, x, m)
    for _ in range(m):
        l = random.randint(1, n)
        r = random.randint(l, min(l + 10**6, n))
        print(l, r)
""",

"cc_536_B. Tavas and Malekas": """\
import random, string
random.seed(42)
n = 500000
pattern_len = 320
p = ''.join(random.choices(string.ascii_lowercase, k=pattern_len))
positions = []
pos = random.randint(1, 100)
while pos + pattern_len - 1 <= n:
    positions.append(pos)
    pos += pattern_len + random.randint(1, 50)
k = len(positions)
print(n, k)
print(p)
if k:
    print(*positions)
""",

"cc_110_C. Lucky Sum of Digits": """\
print(1000000)
""",

"cc_1398_F. Controversial Rounds": """\
import random
random.seed(42)
n = 1000000
s = ''.join(random.choice('01?') for _ in range(n))
print(n)
print(s)
""",

"cc_1223_A. CME": """\
import random
random.seed(42)
q = 1000
print(q)
for _ in range(q):
    print(random.randint(2, 10**9))
""",

"cc_551_D. GukiZ and Binary Operations": """\
import random
random.seed(42)
n = 10**18
k = random.randint(0, 10**18)
l = random.randint(0, 64)
m = random.randint(1, 10**9 + 7)
print(n, k, l, m)
""",

"cc_990_E. Post Lamps": """\
import random
random.seed(42)
n = 1000000
m = 200
k = 1000
block = sorted(random.sample(range(1, n), m))
costs = [random.randint(1, 10**9) for _ in range(k)]
print(n, m, k)
print(*block)
print(*costs)
""",

"cc_p00162 Hamming Numbers": """\
import random
random.seed(42)
for _ in range(50):
    a = random.randint(1, 500000)
    b = random.randint(a, 1000000)
    print(a, b)
print(0)
""",

"cc_1059_C. Sequence Transformation": """\
print(1000000)
""",

"cc_142_A. Help Farmer": """\
print(720720000)
""",

"cc_755_D. PolandBall and Polygon": """\
import random
random.seed(42)
n = 1000000
k = 1
print(n, k)
""",

"cc_1407_E. Egor in the Republic of Dagestan": """\
import random
random.seed(42)
n = 500000
m = 500000
print(n, m)
for _ in range(m):
    u = random.randint(1, n)
    v = random.randint(1, n)
    w = random.randint(0, 1)
    print(u, v, w)
""",

"cc_431_D. Random Task": """\
import random
random.seed(42)
m = random.randint(0, 10**18)
k = random.randint(1, 63)
print(m, k)
""",

"cc_289_C. Polo the Penguin and Strings": """\
import random
random.seed(42)
n = 1000000
k = 26
print(n, k)
""",

"cc_1175_C. Electrification": """\
import random
random.seed(42)
T = 1
n = 200000
k = 100000
pts = sorted(random.randint(-10**9, 10**9) for _ in range(n))
print(T)
print(n, k)
print(*pts)
""",

"cc_852_B. Neural Network country": """\
import random
random.seed(42)
n = 500
l = 1000000
m = 50
print(n, l, m)
print(*[random.randint(0, 10**9) for _ in range(n)])
print(*[random.randint(0, 10**9) for _ in range(n)])
print(*[random.randint(0, 10**9) for _ in range(n)])
""",

"cc_p03961 CODE FESTIVAL 2016 qual C - Encyclopedia of Permutations": """\
import random
random.seed(42)
n = 500000
perm = list(range(1, n + 1))
random.shuffle(perm)
indices = random.sample(range(n), n // 2)
for i in indices:
    perm[i] = 0
print(n)
print(*perm)
""",

"cc_172_D. Calendar Reform": """\
print(1, 10000000)
""",

"cc_954_G. Castle Defense": """\
import random
random.seed(42)
n = 500000
r = 250000
k = 10**9
print(n, r, k)
print(*[random.randint(0, 10**9) for _ in range(n)])
""",

"cc_1353_E. K-periodic Garland": """\
import random
random.seed(42)
T = 1
n = 1000000
k = random.randint(1, n)
s = ''.join(random.choice('01') for _ in range(n))
print(T)
print(n, k)
print(s)
""",

"cc_385_C. Bear and Prime Numbers": """\
import random
random.seed(42)
n = 1000000
xs = [random.randint(2, 10**7) for _ in range(n)]
m = 50000
queries = [(random.randint(2, 10**7), random.randint(2, 10**7)) for _ in range(m)]
queries = [(min(a, b), max(a, b)) for a, b in queries]
print(n)
print(*xs)
print(m)
for l, r in queries:
    print(l, r)
""",

"cc_717_D. Dexterina’s Lab": """\
import random
random.seed(42)
n = 10**9
x = 127
print(n, x)
""",

"cc_p01113 Floating-Point Numbers": """\
import random
random.seed(42)
for _ in range(30):
    n = random.randint(1, 100)
    bits = ''.join(random.choice('01') for _ in range(52))
    print(n)
    print(bits)
print(0)
""",

"cc_372_A. Counting Kangaroos is Fun": """\
import random
random.seed(42)
n = 500000
print(n)
for _ in range(n):
    print(random.randint(1, 100000))
""",

"cc_615_E. Hexagons": """\
print(10**18)
""",

"cc_287_B. Pipeline": """\
import random
random.seed(42)
n = 10**18
k = random.randint(2, 10**9)
print(n, k)
""",


# ── 195 new generators ────────────────────────────────────────────────────────

# ── Single-integer CF problems ────────────────────────────────────────────────

"cc_630_A. Again Twenty Five!": """\
print(10**18)
""",

"cc_393_C. Blocked Points": """\
print(10**9)
""",

"cc_622_A. Infinite Sequence": """\
print(10**18)
""",

"cc_1037_A. Packets": """\
print(10**9)
""",

"cc_1269_A. Equation": """\
print(10**9)
""",

"cc_802_H. Fake News (medium)": """\
print(10**6)
""",

"cc_630_D. Hexagons!": """\
print(10**9)
""",

"cc_630_J. Divisibility": """\
print(10**18)
""",

"cc_630_R. Game": """\
print(10**18)
""",

"cc_177_B1. Rectangular Game": """\
print(10**9)
""",

"cc_669_A. Little Artem and Presents": """\
print(10**9)
""",

"cc_690_A1. Collective Mindsets (easy)": """\
print(10**9 + 1)
""",

"cc_690_A2. Collective Mindsets (medium)": """\
print(10**9 + 1)
""",

"cc_552_B. Vanya and Books": """\
print(10**9)
""",

"cc_876_C. Classroom Watch": """\
print(10**9)
""",

"cc_9_C. Hexadecimal's Numbers": """\
print(10**9)
""",

"cc_959_E. Mahmoud and Ehab and the xor-MST": """\
print(10**9)
""",

"cc_736_B. Taxes": """\
print(998244352)
""",

"cc_735_D. Taxes": """\
print(998244352)
""",

"cc_784_B. Kids' Riddle": """\
print(2 * 10**9)
""",

"cc_535_B. Tavas and SaDDas": """\
print(4444444)
""",

"cc_9_C. Hexadecimal's Numbers": """\
print(10**9)
""",

"cc_1091_C. New Year and the Sphere Transmission": """\
print(10**9)
""",

"cc_776_E. The Holmes Children": """\
print(331725641503, 465583326659)
""",

"cc_854_B. Maxim Buys an Apartment": """\
print(10**9, 1)
""",

"cc_630_P. Area of a Star": """\
import random
random.seed(42)
n = 999999937  # large prime
r = 10**9
print(n, r)
""",

"cc_777_A. Shell Game": """import random
random.seed(42)
n = 10**9
x = random.randint(0, 2)
print(n)
print(x)
""",

# ── Multiple-test-case CF problems ────────────────────────────────────────────

"cc_1374_B. Multiply by 2, divide by 6": """\
import random
random.seed(42)
t = 20000
print(t)
for _ in range(t):
    # Use powers of 6 and 2*3^k forms for interesting cases, plus randoms
    print(random.choice([6**random.randint(0, 12), 2**random.randint(0, 30),
                         3**random.randint(0, 18), random.randint(1, 10**9)]))
""",

"cc_1374_A. Required Remainder": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    x = random.randint(2, 10**9)
    y = random.randint(0, x - 1)
    n = random.randint(y, 10**9)
    print(x, y, n)
""",

"cc_1277_A. Happy Birthday, Polycarp!": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    print(random.randint(1, 10**9))
""",

"cc_1352_B. Same Parity Summands": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    n = random.randint(1, 10**9)
    k = random.randint(1, 100)
    print(n, k)
""",

"cc_1352_C. K-th Not Divisible by n": """\
import random
random.seed(42)
t = 1000
print(t)
for _ in range(t):
    n = random.randint(2, 10**9)
    k = random.randint(1, 10**9)
    print(n, k)
""",

"cc_1487_B. Cat Cycle": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    n = random.randint(2, 10**9) * 2 + 1  # odd n >= 3
    k = random.randint(1, 10**9)
    print(n, k)
""",

"cc_1409_B. Minimum Product": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    x = random.randint(1, 10**9)
    y = random.randint(1, 10**9)
    a = random.randint(x, 10**9)
    b = random.randint(y, 10**9)
    n = random.randint(0, 10**9)
    print(a, b, x, y, n)
""",

"cc_1294_C. Product of Three Numbers": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    # products of 3 distinct factors >= 2
    a = random.randint(2, 1000)
    b = random.randint(2, 1000)
    c = random.randint(2, 1000)
    print(a * b * c)
""",

"cc_1475_B. New Year's Number": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    print(random.randint(1, 10**6))
""",

"cc_451_C. Predict Outcome of the Game": """\
import random
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = random.randint(3, 10**9)
    k = random.randint(1, n)
    a = random.randint(-1, n)
    b = random.randint(-1, n)
    print(n, k, a, b)
""",

"cc_858_A. k-rounding": """\
import random
random.seed(42)
n = random.randint(1, 10**9)
k = random.randint(0, 9)
print(n, k)
""",

# ── String/array CF problems ──────────────────────────────────────────────────

"cc_538_B. Quasi Binary": """\
print(999999)
""",

"cc_219_C. Color Stripe": """\
import random, string
random.seed(42)
n = 500000
k = 26
s = ''.join(random.choice(string.ascii_uppercase[:k]) for _ in range(n))
print(n, k)
print(s)
""",

"cc_1451_C. String Equality": """\
import random, string
random.seed(42)
t = 100
print(t)
for _ in range(t):
    n = 10**5
    k = random.randint(2, n)
    a = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
    b = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
    print(n, k)
    print(a)
    print(b)
""",

"cc_879_C. Short Program": """\
import random
random.seed(42)
n = 5000
print(n)
for _ in range(n):
    op = random.choice(['^', '&', '|'])
    val = random.randint(0, 1023)
    print(op, val)
""",

"cc_835_B. The number on the board": """\
import random
random.seed(42)
k = 10**9
# Large number whose digit sum > k, so answer is 0 changes needed
n_str = '9' * 100000
print(k)
print(n_str)
""",

"cc_604_B. More Cowbell": """\
import random
random.seed(42)
n = 200000
k = 100000
vals = sorted(random.randint(1, 10**9) for _ in range(n))
print(n, k)
print(*vals)
""",

"cc_577_B. Modulo Sum": """\
import random
random.seed(42)
n = 10**6
m = 1000
print(n, m)
print(*[random.randint(1, m) for _ in range(n)])
""",

"cc_985_E. Pencils and Boxes": """\
import random
random.seed(42)
n = 200000
l = 1
r = 200000
print(n, l, r)
vals = sorted(random.randint(1, 10**9) for _ in range(n))
print(*vals)
""",

"cc_659_G. Fence Divercity": """\
import random
random.seed(42)
n = 1000000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_316_D3. PE Lesson": """\
import random
random.seed(42)
n = 200000
print(n)
print(*[random.choice([1, 2]) for _ in range(n)])
""",

"cc_1300_E. Water Balance": """\
import random
random.seed(42)
n = 500000
print(n)
print(*[random.randint(1, 10**6) for _ in range(n)])
""",

"cc_841_B. Godsend": """\
import random
random.seed(42)
n = 100000
print(n)
print(*[random.randint(1, 10**9) for _ in range(n)])
""",

"cc_1191_D. Tokitsukaze, CSL and Stone Game": """\
import random
random.seed(42)
n = 100000
vals = sorted(random.randint(0, 10**9) for _ in range(n))
print(n)
print(*vals)
""",

"cc_939_B. Hamster Farm": """\
import random
random.seed(42)
N = 10**9
K = 50
print(N, K)
vals = [random.randint(1, 10**9) for _ in range(K)]
print(*vals)
""",

"cc_474_B. Worms": """\
import random
random.seed(42)
n = 100000
piles = [random.randint(1, 10**4) for _ in range(n)]
total = sum(piles)
q = 10000
print(n)
print(*piles)
print(q)
for _ in range(q):
    print(random.randint(1, total))
""",

"cc_367_C. Sereja and the Arrangement of Numbers": """\
import random
random.seed(42)
n = 4000
m = 4000
print(n, m)
for _ in range(m):
    # cnt_x and cnt_y - frequencies up to n
    a = random.randint(1, 10**5)
    b = random.randint(1, 10**5)
    print(a, b)
""",

"cc_39_F. Pacifist frogs": """import random
random.seed(42)
n = 200
m = 100
k = 80
print(n, m, k)
print(*[random.randint(1, 200) for _ in range(m)])
print(*[random.randint(1, 200) for _ in range(k)])
""",

"cc_102_D. Buses": """\
import random
random.seed(42)
n = 100000
m = 100000
print(n, m)
for _ in range(m):
    l = random.randint(0, n - 1)
    r = random.randint(l + 1, n)
    print(l, r)
""",

"cc_1095_E. Almost Regular Bracket Sequence": """\
import random
random.seed(42)
n = 500000
s = []
depth = 0
for i in range(n):
    if depth == 0:
        s.append('(')
        depth += 1
    elif i >= n - depth:
        s.append(')')
        depth -= 1
    else:
        c = random.choice('()')
        s.append(c)
        depth += (1 if c == '(' else -1)
while depth > 0:
    s.append(')')
    depth -= 1
print(n)
print(''.join(s))
""",

"cc_820_D. Mister B and PR Shifts": """\
import random
random.seed(42)
n = 200000
perm = list(range(1, n + 1))
random.shuffle(perm)
print(n)
print(*perm)
""",

"cc_67_D. Optical Experiment": """\
import random
random.seed(42)
n = 200000
perm1 = list(range(1, n + 1))
perm2 = list(range(1, n + 1))
random.shuffle(perm1)
random.shuffle(perm2)
print(n)
print(*perm1)
print(*perm2)
""",

"cc_608_E. Marbles": """import random
random.seed(42)
n = 200001
s1 = ''.join(random.choice('NESW') for _ in range(n))
s2 = ''.join(random.choice('NESW') for _ in range(n))
print(n)
print(s1)
print(s2)
""",

"cc_60_E. Mushroom Gnomes": """\
import random
random.seed(42)
a = random.randint(1, 10**9)
b = random.randint(a, a + 10**6)
q = 1000
k = random.randint(q, q + 100)
print(a, b, q, k)
vals = sorted(random.sample(range(a, b + 1), k))
print(*vals)
for _ in range(q):
    print(random.randint(1, k - 1), random.randint(1, k - 1))
""",

"cc_1027_C. Minimum Value Rectangle": """\
import random
random.seed(42)
total = 0
tests = []
while total < 200000:
    n = random.randint(4, min(200000 - total, 1000))
    sticks = sorted(random.randint(1, 10**9) for _ in range(n))
    # make pairs so rectangle always exists
    pairs = []
    for i in range(0, n - 1, 2):
        pairs.extend([sticks[i], sticks[i]])
    if len(pairs) < 4:
        pairs = [1, 1, 2, 2]
    tests.append(pairs)
    total += len(pairs)
    if total >= 200000:
        break
print(len(tests))
for t in tests:
    print(len(t))
    print(*t)
""",

"cc_1096_C. Polygon for the Angle": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    # angle 1/2 degrees (numerator/denominator)
    num = random.randint(1, 179)
    den = random.choice([1, 2])
    print(num)
    print(den)
""",

"cc_1348_D. Phoenix and Science": """\
import random
random.seed(42)
t = 10000
print(t)
for _ in range(t):
    print(random.randint(1, 10**18))
""",

"cc_449_D. Jzzhu and Numbers": """\
import random
random.seed(42)
n = 10**6
print(n)
print(*[random.randint(0, 10**6) for _ in range(n)])
""",

"cc_404_D. Minesweeper 1D": """\
import random
random.seed(42)
n = 2000
s = ''.join(random.choice('*?01') for _ in range(n))
# replace '2' with '?'
# make sure no '2' appears (it's not a valid character except in pattern)
print(s)
""",

"cc_183_B. Zoo": """import random
random.seed(42)
n = 1000
m = 2000
print(n, m)
for _ in range(m):
    print(random.randint(1, 10**9), random.randint(1, 10**9))
""",

"cc_1225_C. p-binary": """\
import random
random.seed(42)
n = 10**9
p = random.randint(-10**9, 10**9)
print(n, p)
""",

"cc_1396_C. Monster Invaders": """\
import random
random.seed(42)
n = 3 * 10**5
S1 = random.randint(1, 10**9)
S2 = random.randint(2 * S1, 3 * S1)
S3 = random.randint(3 * S1, 4 * S1)
k = random.randint(1, 10**6)
print(n, S1, S2, S3, k)
print(*[random.randint(0, 2) for _ in range(n)])
""",

"cc_570_B. Simple Game": """\
import random
random.seed(42)
n = 10**9
m = random.randint(1, n)
print(n, m)
""",

"cc_1493_E. Enormous XOR": """\
import random
random.seed(42)
t = 1
print(t)
n = 1000
l = '0' + ''.join(random.choice('01') for _ in range(n - 1))
r = '1' + ''.join(random.choice('01') for _ in range(n - 1))
print(n)
print(l)
print(r)
""",

"cc_1469_E. A Bit Similar": """\
import random
random.seed(42)
t = 1
print(t)
n = 200000
k = 200
s = ''.join(random.choice('01') for _ in range(n))
print(n, k)
print(s)
""",

"cc_758_F. Geometrical Progression": """\
import random
random.seed(42)
n = 10
l = 1
r = 10**9
print(n, l, r)
""",

"cc_1037_C. Equalize": """\
import random
random.seed(42)
n = 200000
a = ''.join(random.choice('01') for _ in range(n))
b = ''.join(random.choice('01') for _ in range(n))
print(n)
print(a)
print(b)
""",

"cc_236_C. LCM Challenge": """\
print(10**9)
""",

"cc_625_A. Guest From the Past": """\
import random
random.seed(42)
n = 10**18
a = random.randint(2, 10**15)
b = a + random.randint(1, 10**14)
c = random.randint(1, a - 1)
print(n)
print(a)
print(b)
print(c)
""",

"cc_650_B. Image Preview": """\
import random
random.seed(42)
n = 200000
a = random.randint(1, 10**9)
b = random.randint(1, 10**9)
t = random.randint(1, 10**18)
s = ''.join(random.choice('wh') for _ in range(n))
print(n, a, b, t)
print(s)
""",

"cc_834_B. The Festive Evening": """\
import random, string
random.seed(42)
n = 200000
k = 26
s = ''.join(random.choice(string.ascii_uppercase[:k]) for _ in range(n))
print(n, k)
print(s)
""",

"cc_940_B. Our Tanya is Crying Out Loud": """\
import random
random.seed(42)
n = 10**9
k = random.randint(2, 10**9)
A = random.randint(1, 10**9)
B = random.randint(1, 10**9)
print(n)
print(k)
print(A)
print(B)
""",

"cc_779_B. Weird Rounding": """\
import random
random.seed(42)
n = random.randint(10**8, 10**9)
k = random.randint(1, 9)
print(n, k)
""",

"cc_12_B. Correct Solution?": """\
import random
random.seed(42)
# Large number with repeated digit
n1 = '1' * 100000
n2 = '1' * 99999 + '2'
print(n1)
print(n2)
""",

"cc_103_C. Russian Roulette": """\
import random
random.seed(42)
n = 1000000
k = 500000
q = n
print(n, k, q)
for i in range(1, q + 1):
    print(i)
""",

# ── Japanese (cc_p*) problems ─────────────────────────────────────────────────

"cc_p00865 Expected Allowance": """\
import random
random.seed(42)
for _ in range(50):
    n = random.randint(1, 13)
    m = random.randint(2, 2008)
    k = random.randint(-n, n * m)
    print(n, m, k)
print(0, 0, 0)
""",

"cc_p01076 Graph Making": """\
import random
random.seed(42)
for _ in range(20):
    n = random.randint(2, 10**9)
    d = random.randint(1, min(n - 1, 100))
    print(n, d)
""",

"cc_p00184 Tsuruga Castle": """\
import random
random.seed(42)
ages = [random.randint(0, 130) for _ in range(1000)]
for a in ages:
    print(a)
print(0)
""",

"cc_p02468 Power": """\
import random
random.seed(42)
for _ in range(50):
    m = random.randint(1, 100)
    n = random.randint(1, 10**9)
    print(m, n)
""",

"cc_p02116 nCm": """\
import random
random.seed(42)
for _ in range(30):
    print(random.randint(2, 10**18))
""",

"cc_p02470 Euler's Phi Function": """\
import random
random.seed(42)
print(10**9)
""",

"cc_p00158 Collatz's Problem": """\
import random
random.seed(42)
for _ in range(30):
    print(random.randint(1, 10**6))
print(0)
""",

"cc_p01078 Star": """\
import random
random.seed(42)
for _ in range(20):
    while True:
        N = random.randint(5, 200)
        K = random.randint(2, N - 1)
        import math
        if math.gcd(N, K) == 1 and (N % 2 != 0 or K % 2 != 0):
            break
    print(N, K)
""",

"cc_p00099 Surf Smelt Fishing Contest II": """\
import random
random.seed(42)
p = 100
e = 10000
print(p, e)
for _ in range(e):
    participant = random.randint(1, p)
    fish = random.randint(-50, 50)
    print(participant, fish)
""",

"cc_p00748 Pollock's conjecture": """\
import random
random.seed(42)
for _ in range(50):
    print(random.randint(1, 10**6))
print(0)
""",

"cc_p00463 Amidakuji": """\
import random
random.seed(42)
# Multiple test cases, terminated by 0 0 0 -1
for _ in range(5):
    n = random.randint(2, 10)
    m = random.randint(1, 30)
    target = random.randint(1, n)
    prize = random.randint(10, 1000)
    print(n, m, target, prize)
    for i in range(1, n + 1):
        print(prize + random.randint(-100, 100))
    # horizontal bars: position and row
    used = set()
    bars = []
    for _ in range(m):
        for attempt in range(100):
            pos = random.randint(1, n - 1)
            row = random.randint(1, m)
            if (pos, row) not in used and (pos - 1, row) not in used and (pos + 1, row) not in used:
                used.add((pos, row))
                bars.append((pos, row))
                break
    for pos, row in bars:
        print(pos, row)
print(0, 0, 0, -1)
""",

"cc_p00490 Best Pizza": """import sys
print('''3
12 2
200
87
1149
010''', end="")
""",

"cc_p00819 Unreliable Message": """import sys
print('''5
AJMP
aB33d
E
86AE
AM
-1
MPJE
WaEaETC302Q
CP
rTurnAGumdam1isdefferentf''', end="")
""",

"cc_p00914 Equal Sum Sets": """\
import random
random.seed(42)
for _ in range(10):
    n = random.randint(5, 20)
    k = random.randint(2, min(n, 8))
    # generate a valid s
    import itertools
    elems = random.sample(range(1, n + 1), k)
    s = sum(elems)
    print(n, k, s)
print(0, 0, 0)
""",

"cc_p01316 Differential Pulse Code Modulation": """import sys
print('''2 7
-1
2
1
0
-1
-4
-4
131
137
2 7
4
2
1
0
-1
-2
-4
124
123
10 7
-4
-2
-1
0
1
2
7
106
84
135
134
132
128
124
122
121
122
5 1
61
0
-1
0
-1
0
4 1
0
255
-1
255
0
0 0''', end="")
""",

"cc_p01646 Dictionary": """\
import random, string
random.seed(42)
for _ in range(5):
    n = random.randint(2, 8)
    print(n)
    words = set()
    while len(words) < n:
        w = ''.join(random.choice(string.ascii_lowercase[:6]) for _ in range(random.randint(1, 5)))
        words.add(w)
    for w in words:
        print(w)
print(0)
""",

"cc_p00949 Hidden Anagrams": """\
import random, string
random.seed(42)
for _ in range(30):
    letters = list(string.ascii_lowercase[:10])
    n = random.randint(3, 10)
    a = ''.join(random.choice(letters) for _ in range(n))
    b = ''.join(random.choice(letters) for _ in range(n + random.randint(0, 5)))
    print(a)
    print(b)
""",

"cc_p00958 Parallel Lines": """\
import random
random.seed(42)
for _ in range(10):
    n = random.randint(1, 5) * 2  # even
    print(n)
    pts = set()
    while len(pts) < n:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        pts.add((x, y))
    for x, y in pts:
        print(x, y)
""",

"cc_p00822 Weather Forecast": """import sys
print('''1
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 0 1 1 0 0 1
0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1
0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0
0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1
0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0
3
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0
0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0
0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 1 0 -1 0 0 0 0 0
0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0
-1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0
1 0 0 1 1 0 0 0 0 1 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 1 0 1 0 1 0 -1 0 0 0 0
0''', end="")
""",

"cc_p01086 Short Phrase": """import sys
print('''9
do
hte
best
and
enjoy
today
at
acm
icpc
14
oh
yes
by
far
it
is
wow
so
bad
to
me
you
know
hey
15
abcde
fghijkl
mnopq
rstuvwx
yzz
abcde
fghijkl
mnopq
rstuvwx
yz
abcde
fghijkl
mnopq
rstuvwx
yz
0''', end="")
""",

"cc_p01694 Step Aerobics": """\
import random
random.seed(42)
for _ in range(10):
    n = random.randint(2, 30)
    moves = [random.choice(['lu', 'ru', 'ld', 'rd']) for _ in range(n)]
    print(n)
    print(' '.join(moves))
print(0)
""",

"cc_p01684 Venn Diagram": """\
import random
random.seed(42)
for _ in range(20):
    n1 = random.randint(0, 100)
    n2 = random.randint(0, 100)
    a = random.randint(0, min(n1, n2))
    b = random.randint(0, n1)
    c = random.randint(0, n2)
    print(n1, n2, a, b, c)
print(0, 0, 0, 0, 0)
""",

"cc_p00534 Silk Road": """\
import random
random.seed(42)
for _ in range(5):
    N = random.randint(5, 100)
    h = random.randint(5, 100)
    print(N, h)
    for _ in range(N):
        print(random.randint(1, 200))
    for _ in range(N + 1):
        print(random.randint(1, 100))
    for _ in range(N + 1):
        print(random.randint(1, 100))
""",

"cc_p01227 Country Road": """import sys
print('''6
5 2
10 30 40 135 100
7 3
3 4 10 17 21 26 46
1 1
101
2 1
0 1000000
3 5
30 011 211
6 4
0 10 20 30 40 50''', end="")
""",

"cc_p00780 Goldbach's Conjecture": """\
import random
random.seed(42)
for _ in range(30):
    n = random.randint(2, 50000) * 2  # even
    print(n)
print(0)
""",

"cc_p00748 Pollock's conjecture": """\
import random
random.seed(42)
for _ in range(50):
    print(random.randint(1, 10**5))
print(0)
""",

"cc_p01289 Strange Couple": """import sys
print('''5 1 5
0 0 1 0 0
0 2 1 0 0
2 0 0 1 0
1 0 0 1 0
0 1 0 0 1
0 -1 -1 1 0
0 0 0''', end="")
""",

"cc_p00483 Planetary Exploration": """import sys
print('''4 7
2
JIOIOIJ
IJIOJOO
IOOJIOJ
OOJJIJO
3 7 0 7
4 6 3 4
2 3 2 2
1 1 -1 10''', end="")
""",

"cc_p00812 Equals are Equals": """\
import random, string
random.seed(42)
def rand_expr():
    ops = ['+', '-', '*', '^']
    vars_ = list('abcde')
    terms = [random.choice(vars_) for _ in range(random.randint(1, 4))]
    expr = terms[0]
    for t in terms[1:]:
        expr += random.choice(ops) + t
    return expr
for _ in range(5):
    for _ in range(random.randint(2, 5)):
        print(rand_expr())
    print('.')
print('.')
""",

"cc_p01085 Entrance Examination": """import sys
print('''5 2 4
100
90
82
70
65
5 2 4
100
170
80
75
65
3 1 2
5000
6502
3000
4 2 3
10000
10100
8000
8000
4 2 3
10000
10000
10000
8000
5 2 3
100
80
136
60
111
0 0 0''', end="")
""",

"cc_p00776 Encryption System": """\
import random, string
random.seed(42)
for _ in range(10):
    n = random.randint(3, 20)
    s = ''.join(random.choice(string.ascii_lowercase[:8]) for _ in range(n))
    print(s)
print('#')
""",

"cc_p00907 Find the Outlier": """\
import random
random.seed(42)
for _ in range(5):
    deg = random.randint(1, 4)
    n = deg + 3
    x_vals = sorted(random.sample(range(-20, 20), n))
    # correct polynomial values + one outlier
    coeffs = [random.randint(-5, 5) for _ in range(deg + 1)]
    def poly(x):
        return sum(c * x**i for i, c in enumerate(coeffs))
    outlier_idx = random.randint(0, n - 1)
    print(deg)
    for i, x in enumerate(x_vals):
        y = poly(x)
        if i == outlier_idx:
            y += random.choice([-1, 1]) * random.randint(5, 20)
        print(float(y))
print(0)
""",

"cc_p00910 Let There Be Light": """import sys
print('''12 5 4
0 10 0 1
1 3 0 2
1 4 0 2
0 0 0 2
10 0 0 1
3 -1 0 2
5 -1 0 2
10 10 0 15
-1 -10 0 1
10 -10 0 1
-10 -10 0 1
10 10 0 1
0 10 0 240
10 0 0 200
10 -2 0 52
-10 0 0 100
1 0 0 2
-1 1 0
12 5 4
0 10 0 1
1 5 0 2
1 4 0 2
0 0 0 2
10 0 0 1
3 -1 0 2
5 -1 0 2
10 10 0 15
0 -10 0 1
10 -12 0 1
-10 -10 0 1
10 10 0 1
0 10 1 260
10 0 0 200
10 -2 0 52
-10 0 0 100
1 1 0 2
0 0 0
5 1 3
1 2 0 2
-1 12 -1 8
-2 -3 5 6
-2 1 3 3
-5 2 3 5
1 1 2 7
0 0 0
5 1 2
1 2 0 2
-1 8 -1 8
-2 -3 5 6
-2 1 3 3
-4 2 3 5
1 1 2 7
0 0 0
0 0 0''', end="")
""",

"cc_p01314 Sum of Consecutive Integers": """import sys
print('''9
256
0''', end="")
""",

"cc_p00546 Zombie Island": """import sys
print('''13 21 1 1
1001 6000
11
4 2
3 7
2 4
8 8
8 4
2 10
3 4
6 7
7 10
10 11
5 9
11 10
3 6
4 5
1 6
13 12
6 7
8 11
6 13
7 8
2 12''', end="")
""",

"cc_p01493 DisconnectedGame": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(2, 6)
    print(n)
    for i in range(n):
        row = ''.join(random.choice('NO') for _ in range(n))
        print(row)
""",

"cc_p01806 Dice Stamp": """import sys
print('''1
0 0
1 2 3 4 5 6
RRRRBBBBLLLLFFFF
2
0 0
1 1 1 1 1 1
RRR
2 2
100 100 100 100 100 101
FFF
1
1010 -1000
1 2 3 4 5 6
LFRB
4
-3 -4
1 2 3 4 5 6
BBBBBBBB
4 -3
11 12 13 14 15 16
LLLLLLLL
3 4
21 22 23 24 25 26
FFFFFFFF
-4 3
31 32 33 34 35 36
RRRRRRRR
3
-2 -2
9 3 1 1 1 1
RRRRBLLLBRRBLB
0 -3
2 5 2 5 2 1
BBLBBRBB
3 0
10 13 2 10 1 5
LLFLLBLL
0''', end="")
""",

"cc_p00839 Organize Your Train": """import sys
print('''3 5
0W 1W
0W 2W
0W 2E
0E 1E
1E 2E
aabbcdcee
-
-
-
-
bbaadeecc
3 3
0E 1W
1E 2W
2E 0W
aabb
bbcc
aa
bbbb
cc
aaaa
3 4
0E 1W
0E 2E
1E 2W
2E 0W
ababab
-
-
aaabbb
-
-
0 0''', end="")
""",

"cc_p00840 Mobile Computing": """import sys
print('''5
1.562976116968605
3
1
2
2
1.7218698683022837
3
1
4
1
2.1448962816069557
3
1
2
3
3.0434166674596996
4
2
2
1
3
2.5055384306161197
2
1
2
6
2''', end="")
""",

"cc_p00854 And Then There Was One": """\
import random
random.seed(42)
for _ in range(20):
    n = random.randint(2, 10000)
    k = random.randint(1, 100)
    m = random.randint(1, n)
    print(n, k, m)
print(0, 0, 0)
""",

"cc_p00856 Minimal Backgammon": """import sys
print('''10 1 0 0
7 1 0 0
13 0 0 0
6 11 1 1
2
5
7 1 0 6
1
1
3
4
5
6
0 0 0 0''', end="")
""",

"cc_p00874 Cubist Artwork": """import sys
print('''5 5
1 2 3 2 5
1 2 1 4 5
5 5
2 3 3 1 3
4 1 5 3 2
5 5
1 2 3 4 5
3 3 1 4 5
3 3
7 7 7
7 14 7
3 3
2 4 4
4 3 4
4 3
4 0 2 5
0 2 1
4 4
2 8 8 8
2 3 8 3
10 10
9 9 9 9 17 9 9 17 9 9
9 16 9 9 9 9 9 9 9 3
10 9
20 1 20 20 20 20 20 18 20 20
20 20 20 20 7 17 20 20 4
0 0''', end="")
""",

"cc_p00876 Swimming Jam": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(1, 10)
    print(n)
    for _ in range(n):
        start = random.randint(0, 100)
        duration = random.randint(1, 50)
        print(start, duration)
print(0)
""",

"cc_p00884 Membership Management": """\
import random, string
random.seed(42)
def rname():
    return ''.join(random.choice(string.ascii_lowercase[:10]) for _ in range(random.randint(3, 6)))
for _ in range(5):
    n = random.randint(2, 5)
    print(n)
    for _ in range(n):
        group = rname()
        members = [rname() for _ in range(random.randint(1, 4))]
        print(f"{group}:{','.join(members)}.")
print(0)
""",

"cc_p00885 Balloon Collecting": """import sys
print('''2
10 100
100 270
2
12 100
100 280
3
100 150
10 360
40 450
3
100 150
17 360
40 440
2
100 10
50 200
2
100 100
50 110
1
15 10
4
1 10
2 20
3 100
90 200
0''', end="")
""",

"cc_p00887 Awkward Lights": """import sys
print('''1 1 1
0
2 2 1
1 1
1 1
3 2 1
1 0 1
0 1 -1
3 3 1
1 0 0
0 1 0
1 0 1
4 4 2
1 1 0 1
0 0 0 1
1 0 1 1
1 0 0 0
5 5 1
1 1 1 0 1
0 1 0 1 0
1 0 1 0 1
0 1 1 1 0
1 0 1 0 0
5 1 2
0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 1
11 11 3
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 -1 0 0
0 -1 0 -1 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0
11 11 3
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1 0 0 0
0 0 1 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 -1 0 1 1 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 -1 0 0 0
-1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0
13 13 7
0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 -1 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0 -1 0 0
1 0 0 0 0 0 0 0 0 -1 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 -1
0 0 0 0 0 0 0 0 0 0 0 -1 0
0 0 0 0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 -1 0 1 -1 0 0 0 0 0
1 0 0''', end="")
""",

"cc_p00890 Test Case Tweaking": """import sys
print('''3 3 3
1 2 3
2 3 4
1 3 8
12 12 2010
1 2 0
2 3 3000
3 4 0
4 5 1940
5 6 3000
6 12 2010
2 7 100
7 8 200
8 9 506
12 10 400
10 11 500
11 6 512
10 18 1
1 2 7
1 3 2
1 4 16
2 5 0
3 6 10
2 7 2
3 5 10
3 6 3
3 7 10
4 7 6
5 8 10
6 8 2
6 9 11
7 9 3
1 9 16
8 10 8
9 10 1
8 2 0
0 0 0''', end="")
""",

"cc_p00892 Intersection of Two Prisms": """import sys
print('''4 3
7 4
3 3
0 0
3 1
4 4
0 0
8 1
4 4
0 2
30 12
2 12
4 2
15 2
30 15
13 25
1 0
8 5
14 5
21 3
21 13
18 15
11 15
6 5
6 8
11 4
17 12
9 9
15 6
20 10
18 12
3 3
5 5
10 2
10 10
20 2
10 15
10 8
4 4
-133 137
-99 -104
55 -98
36 97
-99 16
-98 -98
113 -99
146 99
0 0''', end="")
""",

"cc_p00896 Weaker than Planned": """import sys
print('''4
A
AND
CAT
DOG
Z XUW ZVX Z YZT.
2
AZ
AY
ZA.
2
AA
BB
CC.
16
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
ONMLKJIHGFEDCBA
A B C D E F G H I J K L M N O ABCDEFGHIJKLMNO.
0''', end="")
""",

"cc_p00906 One-Dimensional Cellular Automaton": """import sys
print('''5 4 1 3 3 0
0 1 2 -1 1
5 7 1 3 2 0
0 1 2 1 1
5 13 1 3 2 11
0 1 2 0 1
5 5 2 0 1 100
0 1 2 0 1
6 2 0 2 3 1000
0 1 2 0 1 4
20 1000 0 2 3 1000000000
0 1 2 0 1 0 1 2 0 1 0 1 2 0 2 0 1 2 0 1
30 2 1 0 1 1000000000
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 -1 0 0 0 0 0 0 0
30 2 2 1 1 1000000000
1 0 -1 0 0 0 0 0 0 0 0 -1 0 0 0 1 0 -1 0 0 0 0 0 -1 0 0 0 -1 0 0
30 5 0 3 1 1000000000
0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 1 0 0 -1 0 0 0 0 1 0 -1 0 0 0
0 0 0 0 0 0''', end="")
""",

"cc_p01093 Selection of Participants of an Experiment": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(2, 50)
    print(n)
    scores = [random.randint(1, 100) for _ in range(n)]
    print(*scores)
print(0)
""",

"cc_p01101 Taro's Shopping": """\
import random
random.seed(42)
for _ in range(10):
    n = random.randint(2, 100)
    budget = random.randint(1, 1000)
    print(n, budget)
    prices = [random.randint(1, 500) for _ in range(n)]
    print(*prices)
print(0, 0)
""",

"cc_p01103 A Garden with Ponds": """import sys
print('''3 3
0 0 2
0 1 2
2 3 1
3 5
0 3 7 3 3
4 1 0 2 3
3 3 4 0 2
7 7
2 1 1 1 1 0 0
1 0 0 0 1 0 0
1 0 1 1 1 1 1
1 0 0 0 2 0 1
1 1 1 1 1 0 1
0 0 2 1 0 1 1
0 0 1 1 1 2 -1
6 6
1 1 1 1 2 2
1 0 0 2 0 2
1 0 -1 2 1 2
3 3 3 15 13 9
3 0 0 18 -1 9
2 6 3 9 9 2
0 0''', end="")
""",

"cc_p01104 Making Lunch Boxes": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(2, 8)
    m = random.randint(1, n)
    print(n, m)
    for _ in range(n):
        ingredients = ''.join(random.choice('01') for _ in range(m))
        print(ingredients)
print(0, 0)
""",

"cc_p01117 Scores of Final Examination": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(3, 20)
    k = random.randint(2, 5)
    print(n, k)
    for _ in range(k):
        scores = [random.randint(-1, 100) for _ in range(n)]
        print(*scores)
print(0, 0)
""",

"cc_p01119 Balance Scale": """import sys
print('''4 2
9 2 7 21
2 9
6 2
1 3 6 12 16 16
2 9
5 2
7 3 6 12 11
2 8
7 5
15 16 88 48 51 75 111
36 54 57 148 102
0 0''', end="")
""",

"cc_p01125 Misterious Gems": """import sys
print('''2
10 11
11 12
2
N 2
E 1
2
10 11
11 12
2
N 2
W 1
3
0 15
5 10
5 15
5
W 20
S 10
N 20
E 10
S 10
0''', end="")
""",

"cc_p01128 Railroad Conflict": """import sys
print('''2
-10 1 10 1
4
-6 2 -2 -2 1 1
-6 -2 -2 2 1 0
12 2 1 -2 0 0
11 -2 2 2 1 1
8 12 -7 -3
8
4 -5 4 -2 1 2
4 -2 4 9 1 0
6 9 6 14 1 1
-5 6 7 6 0 0
1 0 1 10 0 0
-5 0 -5 10 0 1
-7 -1 7 0 0 1
-1 -1 -1 -3 0 1''', end="")
""",

"cc_p01137 Space Coconut Grab": """\
import random
random.seed(42)
for _ in range(30):
    print(random.randint(1, 10000))
print(0)
""",

"cc_p01139 Surrounding Area": """import sys
print('''10 10
.....W....
....W.W...
...W...W..
....W...W.
.....W...W
......W.W.
BBB....W..
..B..BBBBB
..B..B....
..B..B..W.
5 3
.B...
...BB
.....
1 1
.
0 0''', end="")
""",

"cc_p01142 Karakuri Doll": """import sys
print('''5 3
#####
#K.M#
#####
9 5
#########
#.....###
#.###..M#
#K#######
#########
9 5
#########
#K......#
####.####
####M####
#########
9 5
#########
#M......#
####.####
####K####
#########
7 9
#######
#####M#
#####.#
#.....#
#.###.#
#.....#
#.#####
#K#####
#######
7 6
#######
#####.#
##....#
#K..#.#
###M#.#
#######
7 8
#######
##...##
###.#M#
#.....#
#.#...#
#.#..##
#.K####
#######
9 6
#########
###..##.#
##......#
#K....#.#
##..#M#.#
#########
9 6
#########
#.#######
#....#M.#
#.#...#.#
###K#...#
#########
12 7
############
###...####.#
##K#...M##.#
##.....#...#
#........#.#
###..#...#.#
############
23 16
#######################
#########...###########
##########.###.########
##########.....########
##########.#.#.###...##
########.#.#.######.###
########............###
########.###.######.###
############.######.###
#K...........######.###
####.#######.######.###
####.#######.######.###
####.................M#
####.#######.##########
###...#####...#########
#######################
46 16
##############################################
#..............#..############################
#..#..........#.#.###....#...................#
#..#...............#.#.#.###.................#
#...#..#....#...#.###.#......................#
#...#....#....#.#.###.#.#...#....#....#..#...#
#.#........#..........#.#.#....#....#....#...#
#...#...#.......###.#........#.........#...#.#
#.#...#.........###.#..##.......#........#...#
#...#........#..###.#..##.........#........#.#
#...............###.#..##..#.........#...#...#
############.######.#..##....................#
###########K...........#.########.############
###################...............M###########
##################.......#####################
##############################################
0 0''', end="")
""",

"cc_p01146 Princess in Danger": """import sys
print('''2 1 0 1 0 1
0 1 2
3 1 1 2 0 1
2
1 2 1
1 2 1
3 2 1 2 0 1
2
0 2 1
1 2 1
4 4 1 4 1 3
2
0 1 2
1 2 4
0 2 1
3 0 3
5 3 2 6 0 3
1 2
2 1 2
1 0 1
3 4 1
2 4 1
4 1 2
3 0 2
5 1 2 6 0 3
1 2
4 2 4
2 2 1
4 3 1
0 -1 5
1 4 2
2 0 3
0 0 0 0 0 0''', end="")
""",

"cc_p01180 The Closest Circle": """import sys
print('''4
3.307280934588015 0.7642036899017219 2.9975443060138307
3.0890161534289398 0.6780678528661457 4.906860770397891
3.468973691756493 5.995537907012198 0.06334557687080211
2.4219246439893203 3.2220571965352227 5.068471740319732
0''', end="")
""",

"cc_p01267 Luck Manipulator": """import sys
print('''1 5 7 11 10
10
2 10 7 11 10
2 4
2 1 1 256 0
128 255
2 -1 0 1 0
1234 5678
2 1 1 101 0
10 98
2 1 1 100 0
172 99
2 1 1 10000 -1
1 0
4 1 1 10000 0
2 1
0 0 0 0 0''', end="")
""",

"cc_p01278 Voronoi Island": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(2, 8)
    m = random.randint(3, 8)
    print(n, m)
    for _ in range(n):
        x, y = random.randint(-50, 50), random.randint(-50, 50)
        print(x, y)
    # polygon vertices
    import math
    for i in range(m):
        angle = 2 * math.pi * i / m
        x = round(30 * math.cos(angle))
        y = round(30 * math.sin(angle))
        print(x, y)
print(0, 0)
""",

"cc_p01280 Galaxy Wide Web Service": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(1, 5)
    print(n)
    for _ in range(n):
        # h lo hi W
        h = random.randint(0, 23)
        lo = random.randint(0, 1000)
        hi = random.randint(lo, lo + 5000)
        W = random.randint(1, 200)
        print(h, lo, hi, W)
print(0)
""",

"cc_p01281 Tatami": """\
import random
random.seed(42)
for _ in range(5):
    R = random.choice([2, 4, 6, 8, 10])
    C = random.choice([2, 4, 6, 8, 10])
    print(R, C)
print(0, 0)
""",

"cc_p01284 Erratic Sleep Habits": """import sys
print('''2
-1 23
3
1 1
1 1
3 -1
0''', end="")
""",

"cc_p01287 Colored Octahedra": """\
import random
random.seed(42)
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black']
for _ in range(10):
    faces = [random.choice(colors) for _ in range(8)]
    print(' '.join(faces))
""",

"cc_p01288 Marked Ancestor": """\
import random
random.seed(42)
for _ in range(3):
    N = random.randint(5, 20)
    Q = random.randint(3, 10)
    print(N, Q)
    # tree: node i has parent[i] for i > 1
    for i in range(2, N + 1):
        print(random.randint(1, i - 1))
    for _ in range(Q):
        op = random.choice(['Q', 'M'])
        v = random.randint(1, N)
        print(op, v)
print(0, 0)
""",

"cc_p01290 Queen's Case": """import sys
print('''2 2
QE
EA
3 1
AQE
3 1
AQE
5 5
..E..
.###.
A###Q
."##.
..E..
5 1
A.E.Q
5 5
A....
####.
..E..
.####
....Q
0 0''', end="")
""",

"cc_p00837 Book Replacement": """import sys
print('''2 1 1
1
50
2 2 2
1
50
1
60
2 1 2
2
60 61
1
70
4 2 3
3
60 61 62
1
70
2
80 81
3 1 2
3
60 61 62
2
70 60
1 2 5
2
87 95
3
96 71 35
2
68 2
3
3 18 93
2
57 2
2 2 1
5
1 2 1 3 1
0 0 0''', end="")
""",

"cc_p01293 Whist": """import sys
print('''H
4C 8H QS 5D JD KS 8S AH 6H 7H 3S 7S 6D
TC JC JS KD AC QC QD 2H QH 3H 3C 7C 4D
6C 9C AS TD 5H 6S 5S KH TH AD 9S 8D 2D
8C 5C 2S 7D KC 4S TS JH 4H 9H 2C 9D 3D
D
8D 9D 9S QS 4H 5H JD JS 9H 6S TH 6H QH
QD 9C 5S 7S 7H AC 2D KD 6C 3D 8C TC 7C
5D QC 3S 4S 3H 3D 6D KS JC AS 5C H8 TS
4D 4C 8S 2S 2H KC TD JH 2C AH 7D AD KH
#''', end="")
""",

"cc_p01299 Neko's Treasure": """import sys
print('''3
0 -1 100 001
60 100 50
100 110 10
25 75 50
4
0 0 100 000
50 50 50
150 50 50
50 150 50
150 117 50
0''', end="")
""",

"cc_p01315 Moonlight Farm": """import sys
print('''5
elppa 2 2 -1 1 1 0 1 13 1
an`ban 1 0 3 9 2 2 4 1 0
catror 0 2 2 2 1 4 1 18 2
cqniau 1 3 0 2 -1 3 1 -1 1
fghplant 0 4 0 3 2 3 0 110 1
4
enoki 1 3 2 -1 6 3 1 10 1
tpmato 1 0 0 10 0 3 -1 1 1
qotato -1 0 8 4 1 -1 4 7 1
onion 3 4 0 3 3 7 0 10 1
3
a 15 1 1 1 1 1 1 14 5
c 16 -1 2 0 2 1 2 10 2
c 14 0 0 5 2 2 2 10 1
0''', end="")
""",

"cc_p01317 Mr. Rito Post Office": """\
import random
random.seed(42)
for _ in range(3):
    N = random.randint(2, 8)
    M = random.randint(N - 1, N * 2)
    print(N, M)
    edges = set()
    for i in range(1, N):
        edges.add((i, i+1, random.randint(1, 100), random.choice(['L', 'S'])))
    while len(edges) < M:
        u = random.randint(1, N)
        v = random.randint(1, N)
        if u != v:
            edges.add((min(u,v), max(u,v), random.randint(1, 100), random.choice(['L', 'S'])))
    for u, v, w, t in list(edges)[:M]:
        print(u, v, w, t)
    # ports
    q = random.randint(1, N)
    print(q)
    ports = random.sample(range(1, N+1), q)
    print(*ports)
print(0, 0)
""",

"cc_p01339 Alien's Counting": """\
import random
random.seed(42)
for _ in range(5):
    N = random.randint(3, 10)
    M = random.randint(1, N)
    print(N, M)
    for _ in range(M):
        a = random.randint(1, N)
        b = random.randint(1, N)
        while b == a:
            b = random.randint(1, N)
        print(a, b)
""",

"cc_p01437 Infinity Maze": """import sys
print('''3 3 10
E..
.#.
...
5 5 37
####.
.....
.#S#.
...#.
#.##.
5 10 1
#.#..
#....
##.#.
#..S.
#....
5 4 35
.#.#
....
.##.
.#S.
...#
0 0 0''', end="")
""",

"cc_p01448 A Way to Invite Friends": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(2, 20)
    print(n)
    for _ in range(n):
        lo = random.randint(1, 100000)
        hi = random.randint(lo, lo + 100000)
        print(lo, hi)
""",

"cc_p01450 My friends are small": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(2, 30)
    W = random.randint(10, 200)
    print(n, W)
    for _ in range(n):
        print(random.randint(1, W))
""",

"cc_p01451 Roads on Towns": """import sys
print('''5 2
-1 2
0 13
-2 3
-1 1
1 2
0 6
-1 0''', end="")
""",

"cc_p01479 Chicken or the Egg": """\
import random, string
random.seed(42)
for _ in range(20):
    # strings containing 'chicken', 'egg', or mixed
    base = random.choice(['chicken', 'egg', 'chickenoregg'])
    noise = ''.join(random.choice(string.ascii_lowercase[:5]) for _ in range(random.randint(0, 5)))
    pos = random.randint(0, len(noise))
    s = noise[:pos] + base + noise[pos:]
    print(s)
""",

"cc_p01488 TransferTrain": """import sys
print('''2 10
Warsaw Petersburg
3
Kiev Moscow grubsreteP
150 120
3
Moscow Minsk Warsaw
100 150''', end="")
""",

"cc_p01498 King Slime": """import sys
print('''4 3 3
1 1
0 3
3 1
3 3''', end="")
""",

"cc_p01499 Rabbit Game Playing": """import sys
print('''15 11
35
15
4
-4
0''', end="")
""",

"cc_p01624 Ononokomachi's Edit War": """\
import random, string
random.seed(42)
for _ in range(5):
    k = random.randint(1, 5)
    expr = ''.join(random.choice('0123456789+-*/()') for _ in range(random.randint(3, 10)))
    print(k, expr)
print(0, '#')
""",

"cc_p01650 Stack Maze": """import sys
print('''3 3
ac#
b#C
.BA
3 3
aaZ
a#Z
aZZ
3 3
.."
.#.
#..
1 50
abcdefghijklmnopqrstuvwxyYXWVUTSRQPONMLKJIHGFEDCBA
1 50
aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyY
1 50
abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY
1 50
aaaaaaaaaabbbbbbbbbbcccccCCCCCBBBBBBBBBBAAAAAAAAAA
10 10
...#......
a###.#####
.bc...A...
##.#C#d#.#
.#B#.#.###
.#...#e.D.
.#A..###.#
..e.c#..E.
####d###.#
##E...D.C.
0 0''', end="")
""",

"cc_p01670 Medical Inspection": """\
import random
random.seed(42)
for _ in range(5):
    N = random.randint(3, 20)
    D = random.randint(1, 5)
    M = random.randint(1, 5)
    print(N, D, M)
    for _ in range(M):
        a = random.randint(1, N)
        b = random.randint(1, N)
        print(a, b)
""",

"cc_p01676 Tree Reconstruction": """import sys
print('''18 2
1 1
1 3
1 13
2 4
3 0
5 6
0 9
5 12
0 10
6 3
7 9
5 10
9 1''', end="")
""",

"cc_p01677 Broken Audio Signal": """import sys
print('''5
0 x 2 0 x
0
x x
2
1 2
2
-2 0
2
1000000000 x
6
x 0 1 x
-2''', end="")
""",

"cc_p01701 North North West": """import sys
print('''north
west
northwest
northnorthwest
westwestwestnorth
#''', end="")
""",

"cc_p01711 Idempotent Filter": """import sys
print('''00000000111111110000000011111111000000001111111100000000111111110000000011111111000000001111111100000000111111110000000111111111
10000000111111110000000011111111000000001111111100000000111111110000000011111111000000001111111100000000111111110000000011111111
01010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010111010101010101010101010101
#''', end="")
""",

"cc_p01845 Curry Making": """\
import random
random.seed(42)
for _ in range(5):
    N = random.randint(2, 10)
    S = random.randint(1, 5)
    T = random.randint(1, 5)
    U = random.randint(1, 5)
    print(N, S, T, U)
    for _ in range(N):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        c = random.randint(1, 20)
        d = random.randint(1, 20)
        print(a, b, c, d)
print(0, 0, 0, 0)
""",

"cc_p01925 Quiz": """import sys
print('''3 2
5 2 1 3
8 2 2 3
2 3
13 2 1 2
3 1 1
5 1 1
2 5
100 1 1
100 1 1
100 1 1
100 1 1
100 1 1
3 4
5 1 1
5 1 2
100 2 1 3
100 2 2 3
0 0''', end="")
""",

"cc_p01981 Change of the Era Name": """\
import random
random.seed(42)
eras = ['HEISEI', 'SHOWA', 'TAISHO', 'MEIJI']
for _ in range(20):
    era = random.choice(eras)
    y = random.randint(1, 50)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    print(era, y, m, d)
print('#')
""",

"cc_p00713 Circle and Points": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(2, 15)
    print(n)
    for _ in range(n):
        x = round(random.uniform(0, 10), 6)
        y = round(random.uniform(0, 10), 6)
        print(x, y)
print(0)
""",

"cc_p00746 Pablo Squarson's Headache": """import sys
print('''1
5
0 0
0 -1
0 2
0 3
12
0 0
0 0
2 0
3 0
4 1
5 1
6 2
3 2
8 2
9 3
10 3
10
0 2
1 3
2 2
3 2
2 1
5 1
6 1
7 1
3 0
0''', end="")
""",

"cc_p00749 Off Balance": """import sys
print('''4 5
..33
33..
2222
..1.
.111
5 7
....1
.1..1
.1..1
11..1
.2222
..111
...1.
3 6
.3.
233
23.
22.
.11
.11
4 3
2222
..11
..11
4 5
.3..
..33
322.
2211
.11.
3 3
222
2.1
111
3 4
11.
11.
.2/
222
3 4
.11
.11
.2.
222
2 7
11
.1
21
22
23
.3
33
2 3
1.
11
1.0304765877571223
0 0''', end="")
""",

"cc_p00760 Millennium": """import sys
print('''8
1 1 2
344 3 -1
696 5 0
182 9 5
998 8 7
344 2 19
696 4 19
999 10 20''', end="")
""",

"cc_p00764 Chain-Confined Path": """import sys
print('''10
802 0 10
814 0 4
820 1 4
826 1 4
832 3 5
838 9 5
845 7 3
849 10 3
853 14 4
857 18 3
3
0 0 5
8 0 5
8 8 5
3
0 0 5
7 3 6
16 1 5
9
-1 3 5
8 0 8
19 2 11
23 14 6
23 21 6
23 28 6
19 40 8
8 42 8
0 39 5
11
0 0 5
8 0 5
18 8 10
8 16 5
0 16 5
-1 24 5
3 32 5
10 32 5
17 28 8
27 25 3
30 28 5
0''', end="")
""",

"cc_p00775 Vampire": """import sys
print('''4 3
-2 -2 3
1 0 3
1 1 3
2 2
-2 0 8
-2 2 3
6 6
-1 3 1
-4 3 2
-2 4 3
0 -1 0
1 8 3
2 4 1
2 6
-11 6 1
-3 1 2
-6 1 2
-2 0 4
-3 -1 5
-5 -2 6
0 0''', end="")
""",

"cc_p00782 Mobile Phone Coverage": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(2, 6)
    print(n)
    antennas = []
    for _ in range(n):
        x = round(random.uniform(0, 10), 3)
        y = round(random.uniform(0, 10), 3)
        r = round(random.uniform(0.5, 5), 3)
        antennas.append((x, y, r))
        print(x, y, r)
print(0)
""",

"cc_p00815 Life Line": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(2, 5)
    k = random.randint(1, 5)
    print(n, k)
    # triangular board: n rows
    for i in range(1, n + 1):
        row = [random.randint(0, n) for _ in range(i)]
        print(*row)
""",

"cc_p00821 Area of Polygons": """\
import random
random.seed(42)
for _ in range(5):
    n = random.randint(3, 8)
    print(n)
    for _ in range(n):
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        print(x, y)
print(0)
""",

"cc_p00829 Leaky Cryptography": """import sys
print('''8
1 1 1 1 1 1 1 1 8
3 2 3 2 3 2 4 2 10
6 4 4 13 7 b a 2 2e
e1 13 ce 28 ca 6 ab 46 a5d
b08 49e2 6128 f27 8cf2 bc50 7380 7fe1 722b
4eba eb4 a352 fd14 6ac1 fed1 dd06 bb83 392bc
ef593c08 847e522f 74c02b9c 26f3a4e1 e2720a01 6fe66007
7a4e96ad 6ee5cef6 3853cd88
8bf20206 757d6d66 9c3a9525 fbcd7983 82b9571c ddc54bab 853e52da
12047c88 e5524401''', end="")
""",

"cc_p00489 Soccer": """\
import random
random.seed(42)
for _ in range(5):
    N = random.randint(2, 10)
    print(N)
    matches = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g1 = random.randint(0, 5)
            g2 = random.randint(0, 5)
            matches.append((i, j, g1, g2))
    random.shuffle(matches)
    k = random.randint(1, len(matches))
    for i, j, g1, g2 in matches[:k]:
        print(i, j, g1, g2)
    # invalid match to end: -1 sentinel
    print(-1, -1, -1, -1)
print(0)
""",

"cc_p00485 Shopping in JOI Kingdom": """import sys
print('''10 3 1
1 2 5
3 3 2
3 1 1
2''', end="")
""",

"cc_p00492 Illumination": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(2, 10)
    k = random.randint(1, 6)
    print(n, k)
    for _ in range(k):
        row = [random.randint(-1, 2) for _ in range(n)]
        print(*row)
""",

"cc_p00503 Fish": """\
import random
random.seed(42)
for _ in range(3):
    n = random.randint(1, 5)
    q = random.randint(1, 5)
    print(n, q)
    habitats = []
    for _ in range(n):
        x1, y1, z1 = [random.randint(-50, 50) for _ in range(3)]
        x2 = x1 + random.randint(1, 30)
        y2 = y1 + random.randint(1, 30)
        z2 = z1 + random.randint(1, 30)
        print(x1, y1, z1, x2, y2, z2)
    for _ in range(q):
        x, y, z = [random.randint(-60, 80) for _ in range(3)]
        print(x, y, z)
""",

"cc_p00537 Railroad Trip": """import sys
print('''4 4
1 3 2 4
147 131 100
100 50 80
250 70 130''', end="")
""",

"cc_p01103 A Garden with Ponds (dup)": """\
import sys
print('3 3\n0 0 2\n0 1 2\n2 3 1\n3 5\n0 3 7 3 3\n4 1 0 2 3\n3 3 4 0 2\n7 7\n2 1 1 1 1 0 0\n1 0 0 0 1 0 0\n1 0 1 1 1 1 1\n1 0 0 0 2 0 1\n1 1 1 1 1 0 1\n0 0 2 1 0 1 1\n0 0 1 1 1 2 -1\n6 6\n1 1 1 1 2 2\n1 0 0 2 0 2\n1 0 -1 2 1 2\n3 3 3 15 13 9\n3 0 0 18 -1 9\n2 6 3 9 9 2\n0 0', end="")
""",

"cc_p00814 Life Line": """import sys
print('''4 4
   2
  2 3
 1 0 4
1 1 4 0
4 5
   2
  2 3
 3 0 4
1 1 4 0
4 1
   2
  2 3
 3 0 4
1 1 7 0
4 1
   1
  1 1
 1 1 1
1 1 1 0
4 2
   1
  1 0
 1 1 1
1 1 1 0
4 1
   0
  2 2
 5 0 7
0 5 7 0
4 2
   0
  0 3
 1 0 4
0 1 0 4
4 3
   0
  3 3
 3 2 3
0 3 0 3
4 2
   0
  3 3
 3 2 3
0 3 0 3
6 1
     1
    1 2
   1 1 0
  6 7 6 8
 0 7 6 8 2
6 6 7 2 2 0
5 9
    -1
   0 0
  0 0 -1
 0 0 1 0
0 0 0 0 0
5 3
    3
   3 2
  4 3 2
 4 4 0 3
3 3 3 0 3
0 0''', end="")
""",


# ── Additional 6 generators ────────────────────────────────────────────────

"cc_997_B. Roman Digits": """print(10**9 + 1)
""",

"cc_506_E. Mr. Kitayuta's Gift": """import random
random.seed(42)
n = 150
s = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n))
print(s)
""",

"cc_725_B. Food on the Plane": """import random
random.seed(42)
n = 10**18
c = random.choice('abcdef')
print(str(n) + c)
""",

"cc_717_D. Dexterina’s Lab": """import random
random.seed(42)
n = 200
k = 100
probs = [round(random.uniform(0.01, 0.5), 6) for _ in range(n - 2)]
print(n, k)
print(*probs)
""",

"cc_p00224 Bicycle Diet": """import sys
print('''1 1 2 5
13
H L1 18
C1 D 6
C1 H 12
L1 D 10
C1 L1 24
2 2 4 6
100 112
H L1 10
C1 L1 12
C1 D 11
C2 L1 7
C2 D 12
L1 D 8
0 0 0 0''', end="")
""",

"cc_p00915 The Last Ant": """import sys
print('''3 11
R 1
L 2
L 5
1 10
R 1
2 10
R 5
L 7
2 10
R 3
L 8
2 99
R 1
L 98
4 10
L 1
R 2
L 8
R 9
6 10
R 2
R 3
L 4
R 6
L 7
L 8
0 0''', end="")
""",

}


def run_script(code: str, stdin: str = "", timeout: int = 60) -> tuple:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            ["python3", path],
            input=stdin.encode(),
            capture_output=True,
            timeout=timeout,
        )
        return (
            result.stdout.decode("utf-8", errors="replace"),
            result.stderr.decode("utf-8", errors="replace"),
            result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ("", f"TIMEOUT after {timeout}s", 1)
    except Exception as e:
        return ("", str(e), 1)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-only", action="store_true",
                    help="Only run generators for problems that don't have one yet")
    ap.add_argument("--task", type=str, default=None,
                    help="Only run generator for this task_id (partial match ok)")
    args = ap.parse_args()

    cache = json.load(open(CACHE_PATH))
    pool  = {p["task_id"]: p for p in cache["problems"]}

    results = []

    for task_id, gen_code in GENERATORS.items():
        if args.task and args.task not in task_id:
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_id}")

        if task_id not in pool:
            print(f"  NOT IN POOL — skipping")
            continue

        problem = pool[task_id]

        if args.new_only and problem.get("test_case_generator"):
            print(f"  Already has generator — skipping")
            continue

        print(f"  Running generator...")
        stdin, gen_err, gen_rc = run_script(gen_code)

        if gen_rc != 0 or not stdin.strip():
            print(f"  GENERATOR FAILED (rc={gen_rc}): {gen_err[:200]}")
            results.append((task_id, "gen_failed", 0))
            continue

        print(f"  Generator produced {len(stdin)} chars of stdin")

        print(f"  Running reference solution...")
        ref_out, ref_err, ref_rc = run_script(problem["ref_solution"], stdin=stdin)

        if ref_rc != 0:
            print(f"  REF FAILED (rc={ref_rc}): {ref_err[:200]}")
            results.append((task_id, "ref_failed", len(stdin)))
            continue

        print(f"  Ref solution OK — output {len(ref_out)} chars")
        print(f"  Output preview: {repr(ref_out[:80])}")

        problem["test_case_generator"] = gen_code
        results.append((task_id, "ok", len(stdin)))

    json.dump(cache, open(CACHE_PATH, "w"), indent=2)
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    for task_id, status, size in results:
        mark = "✓" if status == "ok" else "✗"
        print(f"  {mark} {task_id[3:45]:42} {status:12} stdin={size}chars")

    saved = sum(1 for _, s, _ in results if s == "ok")
    print(f"\n{saved}/{len(results)} generators stored in cache.")


if __name__ == "__main__":
    main()
