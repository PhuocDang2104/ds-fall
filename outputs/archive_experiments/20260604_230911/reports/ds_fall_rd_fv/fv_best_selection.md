# DS-Fall-RD-FV Best Selection

Selection target: WEDA FP < 12, precision > 0.65, recall >= 0.88, Fall F1 > 0.75, E3 Direction Macro F1 >= 0.86, E7 Direction Macro F1 >= 0.80, E6 BITS Fall F1 >= 0.90, params <= 70k.

## Passing Runs

_No rows._

## Best Trade-off

Best selected run: `FV1_LITE10_AND`.

- E7 WEDA Fall F1: 0.7500
- E7 WEDA precision/recall: 0.6154/0.9600
- E7 WEDA FP/FN: 15/1
- E7 WEDA Direction Macro F1: 0.8390
- E3 Direction Macro F1: 0.8967
- Params: 65970

## Not Run / Failed

| run_id | status | reason |
| --- | --- | --- |
| FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL | not_run | Skipped because safe fall-head unfreezing would require rebuilding a trainable temporal FV graph with explicit fall-logit exposure and verifying unchanged direction outputs. This frozen-probability FV script completed FV1-FV5 and records FV6 as not run instead of applying an unsafe partial unfreeze. |
