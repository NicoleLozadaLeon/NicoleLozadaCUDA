#!/usr/bin/env python3
"""Check an event log and the final summary against the rules of the brief (sections 4, 6, 7).
usage: python3 tools/check_log.py <events.log> <summary.json> <config.json>
       (summary.json = the last line printed by delivery_sim on stdout)
Prints OK, or every broken rule, and exits 1 if any rule is broken.
"""
import json
import sys
from collections import defaultdict
ORDER_EVENTS = {"orderCreated", "orderAssigned", "pickupStarted", "orderPickedUp",
                "orderDelivered", "orderRejected", "orderReassigned"}
def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    log_path, summary_path, config_path = sys.argv[1:4]
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    capacity = cfg["fleet"]["bagCapacity"]
    slots = {r["id"]: r["pickupSlots"] for r in cfg["restaurants"]}
    errors = []
    state = {}                              # order -> created | assigned | pickedUp | delivered | rejected
    last_t = {}                             # order -> t of its last event
    holder = {}                             # order -> courier whose bag holds it
    bags = defaultdict(set)                 # courier -> orders in the bag
    collecting = defaultdict(lambda: defaultdict(int))   # restaurant -> courier -> started, not picked
    count = defaultdict(int)
    per_courier = defaultdict(int)
    stopping_line = None
    def err(n, text):
        if len(errors) < 50:
            errors.append(f"line {n}: {text}")
    with open(log_path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                err(n, "not a valid JSON object")
                continue
            ev, t = e.get("event"), e.get("t")
            if not isinstance(t, int):
                err(n, "t is missing or not an integer")
            if stopping_line is not None:
                err(n, f"{ev} after simulationStopping")
            o, c = e.get("order"), e.get("courier")
            if ev in ORDER_EVENTS:
                if o is None:
                    err(n, f"{ev} without order")
                    continue
                if o in last_t and t < last_t[o]:
                    err(n, f"time runs backwards for {o}")
                last_t[o] = t
            count[ev] += 1
            if ev == "orderCreated":
                if o in state:
                    err(n, f"{o} created twice")
                state[o] = "created"
            elif ev == "orderRejected":
                if state.get(o) not in ("created",):
                    err(n, f"{o} rejected in state {state.get(o)}")
                if e.get("reason") not in ("queueFull", "unreachable", "noCourier"):
                    err(n, f"bad reason {e.get('reason')}")
                state[o] = "rejected"
            elif ev == "orderAssigned":
                if state.get(o) != "created":
                    err(n, f"{o} assigned in state {state.get(o)}")
                state[o] = "assigned"
                holder[o] = c
                bags[c].add(o)
                if len(bags[c]) > capacity:
                    err(n, f"{c} holds {len(bags[c])} orders > bagCapacity {capacity}")
            elif ev == "orderReassigned":
                src, dst = e.get("from"), e.get("to")
                if holder.get(o) != src:
                    err(n, f"{o} reassigned from {src} but held by {holder.get(o)}")
                bags[src].discard(o)
                bags[dst].add(o)
                holder[o] = dst
                if len(bags[dst]) > capacity:
                    err(n, f"{dst} holds {len(bags[dst])} orders > bagCapacity {capacity}")
            elif ev == "pickupStarted":
                r = e.get("restaurant")
                if state.get(o) != "assigned" or holder.get(o) != c:
                    err(n, f"pickupStarted for {o} by {c} in state {state.get(o)}")
                collecting[r][c] += 1
                if len(collecting[r]) > slots.get(r, 0):
                    err(n, f"{len(collecting[r])} couriers collecting at {r} > {slots.get(r)} slots")
            elif ev == "orderPickedUp":
                r = e.get("restaurant")
                if state.get(o) != "assigned" or holder.get(o) != c:
                    err(n, f"orderPickedUp for {o} by {c} in state {state.get(o)}")
                state[o] = "pickedUp"
                collecting[r][c] -= 1
                if collecting[r][c] <= 0:
                    del collecting[r][c]
            elif ev == "orderDelivered":
                if state.get(o) != "pickedUp" or holder.get(o) != c:
                    err(n, f"{o} delivered by {c} in state {state.get(o)} held by {holder.get(o)}")
                state[o] = "delivered"
                bags[c].discard(o)
                per_courier[c] += 1
            elif ev == "simulationStopping":
                stopping_line = n
            elif ev != "courierBreakdown":
                err(n, f"unknown event {ev}")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.loads(f.read().strip().splitlines()[-1])
    delivered = sum(1 for s in state.values() if s == "delivered")
    rejected = sum(1 for s in state.values() if s == "rejected")
    expected = {
        "created": len(state),
        "delivered": delivered,
        "rejected": rejected,
        "pending": len(state) - delivered - rejected,
        "reassigned": count["orderReassigned"],
        "breakdowns": count["courierBreakdown"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary {key} = {summary.get(key)}, log says {value}")
    if summary["created"] != summary["delivered"] + summary["rejected"] + summary["pending"]:
        errors.append("summary: created != delivered + rejected + pending")
    if sum(summary["deliveredPerCourier"].values()) != summary["delivered"]:
        errors.append("summary: deliveredPerCourier does not add up to delivered")
    for courier, n in per_courier.items():
        if summary["deliveredPerCourier"].get(courier) != n:
            errors.append(f"summary: deliveredPerCourier[{courier}] != {n}")
    if stopping_line is None:
        errors.append("no simulationStopping line")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"OK: {len(state)} orders, {delivered} delivered, {rejected} rejected, "
          f"{expected['pending']} pending, {expected['reassigned']} reassigned, "
          f"{expected['breakdowns']} breakdowns")
if __name__ == "__main__":
    main()