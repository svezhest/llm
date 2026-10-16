"""Event sim of N agent episodes round-robin through one serialized prefill
queue + a prompt cache with a byte budget; compare eviction policies
(lru = today's order; largest / mru / smallest = the order among live entries
after stale ones go LRU; wanted = entries of queued episodes evicted last).
Run: python3 evict_sim.py (pure Python, ~1 min). `python3 evict_sim.py clock`:
the round clock as the worker keeps it since the review fix (ticks at every
lookup and every store, stale past 2 x entries; + wall-clock age, 300 / 120 s,
entries a queued request wants exempt) vs the store-only clock.
"""
import heapq, random, statistics, sys

KV = 32 * 1024            # bytes / token (9B fp16 KV)
ST = 49e6                 # one GDN state
NST = 5                   # states per chain entry (chain policy: sys + 3 marks + end)
RATE = 300.0              # prefill tok/s
def size(tokens): return tokens * KV + NST * ST

def run(policy, n=8, gb=8.0, seed=0, hours=3.0, wanted_on=True, K_mul=1, lookup_ticks=False,
        stale_s=0.0):
    rnd = random.Random(seed)
    # episodes
    def new_ep():
        return dict(L=rnd.randint(4000, 6000), turns=rnd.randint(14, 50), t=0)
    eps = [new_ep() for _ in range(n)]
    avg = size(5000 + 0.5 * 30 * 430)      # mean entry over an episode
    budget = gb * 2**30
    cache = {}                # ep -> dict(bytes, used, stored)
    stores = 0
    now = 0.0
    ev = []                   # (time, kind, ep)
    for i in range(n):
        heapq.heappush(ev, (rnd.uniform(0, 20), 'arrive', i))
    queue, busy = [], False
    misses = hits = reprefill = 0
    lat, arr = [], {}
    def victim(keep, waiting):
        cand = [e for e in cache if e != keep]
        if not cand: return None
        K = len(cache) * K_mul * (2 if lookup_ticks else 1)
        def key(e):
            c = cache[e]
            w = wanted_on and e in waiting
            # wall-clock age: only an entry nobody waits for is idle
            stale = stores - c['stored'] > K or (stale_s > 0 and not w and now - c['used'] > stale_s)
            if policy == 'lru': return (w, c['used'])
            if stale: return (w, 0, c['used'])
            if policy == 'largest': return (w, 1, -c['bytes'])
            if policy == 'mru': return (w, 1, -c['used'])
            if policy == 'smallest': return (w, 1, c['bytes'])
            raise ValueError
        return min(cand, key=key)
    while ev:
        now, kind, i = heapq.heappop(ev)
        if now > hours * 3600: break
        if kind == 'arrive':
            arr[i] = now; queue.append(i)
        elif kind == 'prefilled':
            busy = False
            e = eps[i]
            stores += 1
            cache[i] = dict(bytes=size(e['L']), used=now, stored=stores)
            waiting = set(queue)
            while sum(c['bytes'] for c in cache.values()) > budget:
                v = victim(i, waiting)
                if v is None: break
                del cache[v]
            dec = rnd.uniform(10, 20)
            lat.append(now + dec - arr[i])
            heapq.heappush(ev, (now + dec + rnd.uniform(0.3, 2.0), 'next', i))
        elif kind == 'next':
            e = eps[i]
            e['t'] += 1
            if e['t'] >= e['turns']:
                eps[i] = new_ep()          # old entry becomes stale
                cache.pop(('old', i), None)
                if i in cache:            # rename: the finished episode's entry stays as stale
                    cache[('old', i, now)] = cache.pop(i)
            else:
                e['L'] += int(rnd.lognormvariate(6.06, 0.9)) + 50
            heapq.heappush(ev, (now, 'arrive', i))
        if not busy and queue:
            i = queue.pop(0)
            e = eps[i]
            if lookup_ticks:
                stores += 1                # the clock: lookups tick too
            if i in cache:
                hits += 1; cache[i]['used'] = now
                if lookup_ticks:
                    cache[i]['stored'] = stores   # a hit refreshes e.seq (worker)
                suffix = int(rnd.lognormvariate(5.93, 1.0)) + 60
                t = suffix / RATE
            else:
                misses += 1; reprefill += e['L']; t = e['L'] / RATE
            busy = True
            heapq.heappush(ev, (now + t, 'prefilled', i))
    return dict(done=len(lat), policy=policy, misses=misses, hits=hits, miss_rate=misses / max(1, hits + misses),
                reprefill_Mtok=reprefill / 1e6, lat_mean=statistics.mean(lat), lat_p90=sorted(lat)[int(.9 * len(lat))])

if __name__ == '__main__' and sys.argv[1:] == ['clock']:
    # the worker's clock: lookups tick too (2 x entries) + wall-clock age
    # (QWEN_PROMPT_CACHE_STALE_S; 0 = off)
    for gb in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        for p in ('lru', 'mru'):
            for lt, ss in ((False, 0.0), (True, 0.0), (True, 300.0), (True, 120.0)):
                if p == 'lru' and (lt or ss):
                    continue                    # LRU ignores the clock
                rs = [run(p, gb=gb, seed=s, lookup_ticks=lt, stale_s=ss) for s in range(8)]
                m = statistics.mean(r['miss_rate'] for r in rs)
                d = statistics.mean(r['done'] for r in rs)
                l = statistics.mean(r['lat_mean'] for r in rs)
                clock = ('lookup+store' if lt else 'store') + (f'+{ss:.0f}s' if ss else '')
                print(f"gb={gb} {p:4s} clock={clock:17s} done={d:6.0f} miss={m:6.1%} lat={l:6.1f}s")
elif __name__ == '__main__':
    for gb in (3.0, 4.0, 5.0, 6.0, 8.0):
        for wanted in (False, True):
            for p in ('lru', 'largest', 'mru', 'smallest'):
                rs = [run(p, gb=gb, seed=s, wanted_on=wanted) for s in range(8)]
                m = statistics.mean(r['miss_rate'] for r in rs)
                d = statistics.mean(r['done'] for r in rs)
                rp = statistics.mean(r['reprefill_Mtok'] for r in rs)
                l = statistics.mean(r['lat_mean'] for r in rs)
                print(f"gb={gb} wanted={int(wanted)} {p:9s} done={d:6.0f} miss={m:6.1%} reprefill={rp:5.2f}M lat={l:6.1f}s")
