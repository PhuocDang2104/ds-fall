# DS-Fall-RD Data Quality Report

- Feature set: `tilt12`
- Windows: 6414
- Supervised direction samples: 802
- Missing direction labels: 5612
- NaN/Inf before feature engineering: {'nan': 0, 'inf': 0}
- NaN/Inf after feature engineering: {'nan': 0, 'inf': 0}

## Impact Index

```text
{'num_fall_windows': 883, 'early': 20, 'valid': 849, 'late': 14, 'early_pct': 2.2650056625141564, 'valid_pct': 96.14949037372594, 'late_pct': 1.5855039637599093}
```

## Distributions

```text
windows_by_dataset:
dataset
bits       3557
weda       1879
umafall     978

fall_distribution:
fall_label
0    5531
1     883

direction_distribution:
direction_label
none        5531
forward      318
lateral      244
backward     240
other         81
```

## Raw6 Channel Stats

```text
channel         min        max      mean       std
     ax  -78.443626  78.426865 -3.376442  7.230830
     ay  -78.416328  78.419685 -2.763965  5.144765
     az  -78.446022  78.426384  2.604334  4.949934
     gx -255.999985 254.607422 -0.546418 30.719702
     gy -256.877197 253.992203 -0.284923 27.091391
     gz -256.237335 254.650421  0.018414 28.455524
```

## Feature Channel Stats

```text
   channel          min         max      mean        std
        ax   -78.443626   78.426865 -3.376442   7.230830
        ay   -78.416328   78.419685 -2.763965   5.144765
        az   -78.446022   78.426384  2.604334   4.949934
        gx  -255.999985  254.607422 -0.546418  30.719702
        gy  -256.877197  253.992203 -0.284923  27.091391
        gz  -256.237335  254.650421  0.018414  28.455524
   acc_mag     0.002396  134.828857  9.453115   6.302097
  gyro_mag     0.000100  443.404999 15.452473  47.423218
      jerk -3656.198486 4733.454102  0.064337 133.062012
      roll    -3.141544    3.141593 -0.572996   1.079750
     pitch    -1.570783    1.570786  0.418060   0.815795
tilt_delta     0.000000    3.976960  0.128225   0.311039
```

## Peak Summary

```text
       acc_mag_peak  gyro_mag_peak
count   6414.000000    6414.000000
mean      19.786943      41.354614
std       18.929651      98.621620
min        0.991005       0.000000
25%        9.952241       1.084491
50%       11.996794       3.874271
75%       23.426510       9.833360
max      134.828857     443.404999
```

## Warnings

- No warnings generated.
