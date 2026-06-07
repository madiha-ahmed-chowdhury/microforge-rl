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

"cc_717_D. Dexterina's Lab": """\
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
