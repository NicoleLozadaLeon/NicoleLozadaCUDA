#!/usr/bin/env python3
"""Writes config/tests/invalid_*.json, each one with a single defect."""
import copy, json

with open("config/equipetrol.json", encoding="utf-8") as f:
    good = json.load(f)

def write(name, cfg=None, raw=None):
    with open(f"config/tests/{name}.json", "w", encoding="utf-8") as f:
        if raw is not None:
            f.write(raw)
        else:
            json.dump(cfg, f, ensure_ascii=False)

write("invalid_malformed", raw='{"map": {"image": ')
write("invalid_trailing_garbage", raw=json.dumps(good, ensure_ascii=False) + "\ngarbage\n")
c = copy.deepcopy(good); del c["fleet"]; write("invalid_missing_property", c)
c = copy.deepcopy(good); c["streets"][0]["from"] = "nope"; write("invalid_street_node", c)
c = copy.deepcopy(good); c["restaurants"][0]["node"] = "nope"; write("invalid_restaurant_node", c)
c = copy.deepcopy(good); c["nodes"] = []; write("invalid_no_nodes", c)
c = copy.deepcopy(good); c["fleet"]["couriers"] = 0; write("invalid_couriers", c)
c = copy.deepcopy(good); c["incidents"]["breakdownProbability"] = 1.5; write("invalid_probability", c)
c = copy.deepcopy(good); c["fleet"]["startNode"] = "nope"; write("invalid_start_node", c)
c = copy.deepcopy(good); c["nodes"].append(dict(c["nodes"][0])); write("invalid_duplicate_node", c)
print("written 10 invalid configs")
