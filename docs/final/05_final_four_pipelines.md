# Final Four Pipelines

| Pipeline | Fall decision | Direction decision | Feature/Input | Role |
| --- | --- | --- | --- | --- |
| A5_reference | A5 fall head | A5 direction head | tilt12 sequence | deep multitask reference |
| Hybrid_NoTiming | GradientBoosting + FallNoTiming_Full | A5 direction head | summary + tilt12 | practical main |
| EdgeLite_Hybrid | RandomForest + FallNoTiming_Core | A5 direction head | compact summary + tilt12 | compact ablation |
| ML_only_separated | GradientBoosting + FallNoTiming_Full | GradientBoosting + DirectionFullNoTiming | summary only | analysis pipeline |

ML_only_separated is analysis-only unless its direction expert beats A5
consistently across E1-E7.

## Complexity And Edge

| pipeline | fall_source | direction_source | deep_params | a5_fp32_kb | a5_int8_kb | fall_feature_count | direction_feature_count | fall_tree_count | fall_tree_nodes | fall_tree_leaves | direction_tree_count | direction_tree_nodes | direction_tree_leaves | total_tree_count | total_tree_nodes | estimated_complexity_units | edge_suitability | complexity_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_reference | A5_deep_fall_head | A5_direction_head | 65959 | 257.6523 | 64.4131 | 12 | 12 | 0 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 65959 | medium | Single compact temporal CNN; requires TensorFlow/TFLite for deployment. |
| Hybrid_NoTiming | GradientBoosting_FallNoTiming_Full | A5_direction_head | 65959 | 257.6523 | 64.4131 | 132 | 12 | 100 | 1408 | 754.0000 | 0 | 0 | 0.0000 | 100 | 1408 | 67367 | medium | GB fall expert plus A5 direction expert; best practical accuracy, moderate edge cost. |
| EdgeLite_Hybrid | RandomForest_FallNoTiming_Core | A5_direction_head | 65959 | 257.6523 | 64.4131 | 16 | 12 | 300 | 91018 | 45659.0000 | 0 | 0 | 0.0000 | 300 | 91018 | 156977 | medium-high | Compact 16-feature RF fall expert; full pipeline still uses A5 for direction. |
| ML_only_separated | GradientBoosting_FallNoTiming_Full | GradientBoosting_DirectionFullNoTiming | 0 | 0.0000 | 0.0000 | 132 | 132 | 100 | 1408 | 754.0000 | 300 | 4404 | 2352.0000 | 400 | 5812 | 5812 | high | Summary-only tree ensembles; no temporal deep model, CPU-friendly but less direction-consistent. |
