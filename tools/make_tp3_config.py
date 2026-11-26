#!/usr/bin/env python3
"""Write the physically padded TP3 config.json from DeepSeek's own config.

heads 64 -> 72, o_groups 8 -> 9, vocab 129,280 -> 129,408 (all divisible by 3
and still 128-aligned); the real sizes stay in the tensorfold_tp3 marker so the
sampler and tokenizer keep the logical vocabulary.

usage: make_tp3_config.py SRC_CONFIG OUT_CONFIG
"""
import json
import sys

src, out = sys.argv[1:3]
c = json.load(open(src))
t = c.get("text_config", c)
t["vocab_size"] = 129408
t["num_attention_heads"] = 72
t["o_groups"] = 9
c["tensorfold_tp3"] = {
    "schema": 1,
    "world_size": 3,
    "logical_vocab_size": 129280,
    "logical_num_attention_heads": 64,
    "logical_o_groups": 8,
    "padding_group_source": 0,
    "requires_logical_vocab_mask": True,
}
with open(out, "w") as f:
    json.dump(c, f, indent=2)
    f.write("\n")
print("wrote", out)
