# Raw Dataset Technical Notes for DS-Fall

Tai lieu nay mo ta 3 dataset dang co trong `data/raw/` va cach loc/chuan hoa de phu hop voi DS-Fall: mo hinh wrist-worn 6-axis IMU, dau vao chinh `X in R^(100 x 6)` gom:

`[ax, ay, az, gx, gy, gz]`

Trong thiet ke hien tai, 100 timestep tuong ung 2 giay o 50 Hz. Vi vay dataset phu hop nhat la dataset co smartwatch/wrist IMU 50 Hz, co accelerometer + gyroscope, va co nhan huong nga ro rang.

## 1. Tong Quan Nhanh

| Dataset local | Loai file | Quy mo local | Sampling rate | Sensor dung cho DS-Fall | Muc do phu hop |
| --- | --- | ---: | --- | --- | --- |
| `data/raw/WEDA-FALL-main` | CSV tach theo sensor | 15,348 CSV, gom 15,347 sensor file + `fall_timestamps.csv`; 969 logical trials moi frequency | 50 Hz goc, co ban 40/25/10/5 Hz da downsample | `accel_x/y/z` + `gyro_x/y/z` | Tot nhat lam dataset chinh |
| `data/raw/HR_IMU_falldetection_dataset-master` | MATLAB `.mat` | 349 trial files, 21 subjects | 50 Hz | `ax/ay/az` + `droll/dpitch/dyaw` | Tot lam dataset phu/benchmark |
| `data/raw/Dataset` | CSV tron nhieu sensor trong mot file | 984 CSV = 41 users x 24 activities | Paper goc: motion 20 Hz, heart-rate 1 Hz | nen dung `acg` hoac `acc` + `gyro` | Tot lam external/cross-domain, can xu ly ky |

Khuyen nghi uu tien:

1. Train baseline tren WEDA-FALL 50 Hz.
2. Them HIFD de tang da dang huong nga va kiem tra generalization 50 Hz.
3. Dung `Dataset`/BITS-2 cho external validation hoac ablation, vi sampling rate khac va timestamp local khong dang tin cay.

## 2. WEDA-FALL

### 2.1. Duong dan va cau truc

Path:

`data/raw/WEDA-FALL-main`

Cau truc chinh:

```text
dataset/
  fall_timestamps.csv
  50Hz/
    F01/...csv
    ...
    D11/...csv
  40Hz/
  25Hz/
  10Hz/
  5Hz/
```

Moi file sensor co format ten:

`U<user_id>_R<trial_id>_<sensor_type>.csv`

Vi du:

`dataset/50Hz/F07/U02_R03_accel.csv`

nghia la user 02, trial 03, activity `F07`, sensor accelerometer, sampling set 50 Hz.

### 2.2. Quy mo local

Thong ke tu thu muc local:

| Thanh phan | So luong |
| --- | ---: |
| Tong CSV | 15,348 |
| Sensor CSV, khong tinh `fall_timestamps.csv` | 15,347 |
| Logical trials moi frequency | 969 |
| Frequencies | 5 Hz, 10 Hz, 25 Hz, 40 Hz, 50 Hz |
| Fall timestamp rows | 350 |
| Sensor base moi trial | `accel`, `gyro`, `orientation` |
| Sensor phu chi co o 50 Hz | `vertical_accel` |

Tat ca 969 logical trials o moi frequency co du 3 sensor base: accelerometer, gyroscope, orientation.

### 2.3. Sensor va cot du lieu

File `accel`:

```text
accel_time_list,accel_x_list,accel_y_list,accel_z_list
```

File `gyro`:

```text
gyro_time_list,gyro_x_list,gyro_y_list,gyro_z_list
```

File `orientation`:

```text
orientation_time_list,orientation_s_list,orientation_i_list,orientation_j_list,orientation_k_list
```

File `vertical_accel` chi co o 50 Hz:

```text
vertical_Accel_x,vertical_Accel_y,vertical_Accel_z,accel_time_list
```

Voi DS-Fall baseline, chi lay:

`[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`

`orientation` va `vertical_accel` nen bo qua trong baseline 6-axis de giu dung ly thuyet mo hinh. Co the giu chung lam ablation sau.

### 2.4. Activity va nhan

Fall activities:

| Code | Activity | Direction label de xuat |
| --- | --- | --- |
| `F01` | Fall forward while walking caused by a slip | `forward` |
| `F02` | Lateral fall while walking caused by a slip | `lateral` |
| `F03` | Fall backward while walking caused by a slip | `backward` |
| `F04` | Fall forward while walking caused by a trip | `forward` |
| `F05` | Fall backward when trying to sit down | `backward` |
| `F06` | Fall forward while sitting | `forward` |
| `F07` | Fall backward while sitting | `backward` |
| `F08` | Lateral fall while sitting | `lateral` |

ADL activities:

| Code | Activity | Ghi chu |
| --- | --- | --- |
| `D01` | Walking | normal ADL |
| `D02` | Jogging | hard negative |
| `D03` | Walking up and downstairs | hard negative |
| `D04` | Sitting on a chair, wait, get up | posture transition |
| `D05` | Attempt to get up and collapse into chair | near-fall-like hard negative |
| `D06` | Crouching, tie shoes, get up | posture transition |
| `D07` | Stumble while walking | near-fall-like hard negative |
| `D08` | Gently jump without falling | impact-like hard negative |
| `D09` | Hit table with hand | wrist impact hard negative |
| `D10` | Clapping hands | wrist impact hard negative |
| `D11` | Opening and closing door | wrist motion ADL |

User groups:

| Group | Users | Vai tro |
| --- | --- | --- |
| Young participants | `U01` to `U14` | co fall va ADL |
| Elder participants | `U21` to `U31` | chi ADL an toan, khong co fall |

Khong nen chia train/test ngau nhien theo window, vi nhu vay cung subject se bi leak giua train va test. Nen split theo `user_id`.

### 2.5. Fall timestamp

`dataset/fall_timestamps.csv` co format:

```text
filename,start_time,end_time
F01/U01_R01,4.7,7.7
```

`start_time` va `end_time` danh dau khoang actual fall theo giay. WEDA phu hop nhat voi DS-Fall vi co the cat cua so quanh vung fall thay vi lay ca trial dai.

De xuat cat window fall:

1. Load `accel` va `gyro` trong `dataset/50Hz`.
2. Align theo time hoac resample ve grid 50 Hz chung.
3. Trong khoang `start_time` den `end_time`, tim peak cua acceleration magnitude:
   `sqrt(ax^2 + ay^2 + az^2)`.
4. Cat window 2 giay, 100 samples, quanh peak. Neu peak qua gan bien, pad hoac dich window de du 100 samples.

De xuat cat window ADL:

1. Dung sliding window 2 giay, stride 1 giay.
2. Cap so window ADL moi subject/activity de tranh ADL ap dao fall.
3. Giu lai cac hard negative `D05`, `D07`, `D08`, `D09`, `D10` vi chung giong fall hon cac ADL binh thuong.

## 3. HIFD / HR_IMU Fall Detection Dataset

### 3.1. Duong dan va cau truc

Path:

`data/raw/HR_IMU_falldetection_dataset-master`

Cau truc:

```text
subject_01/
  fall/
    fall1.mat
    ...
  non-fall/
    walk.mat
    ...
subject_02/
...
```

### 3.2. Quy mo local

Thong ke tu thu muc local:

| Thanh phan | So luong |
| --- | ---: |
| Subjects | 21 |
| `.mat` files | 349 |
| Fall files | 104 |
| Non-fall files | 245 |
| Length moi trial | min 787, median 1057, max 2578 samples |
| Sampling rate | 50 Hz |

So file theo fall type:

| Fall id | So files local |
| --- | ---: |
| `fall1` | 21 |
| `fall2` | 21 |
| `fall3` | 21 |
| `fall4` | 12 |
| `fall5` | 13 |
| `fall6` | 16 |

Luu y: `fall4`, `fall5`, `fall6` khong du 21 subjects trong local copy. Khi train direction, can tinh class weight hoac balance theo direction.

So file theo non-fall scenario:

| Scenario | So files local |
| --- | ---: |
| `bed` | 19 |
| `chair` | 21 |
| `clap` | 21 |
| `cloth` | 21 |
| `eat` | 13 |
| `hair` | 15 |
| `shoe` | 21 |
| `stair` | 19 |
| `teeth` | 21 |
| `walk` | 21 |
| `wash` | 21 |
| `write` | 11 |
| `zip` | 21 |

README ghi dataset co 19 scenarios: 6 falls, 9 ADLs, 4 near-falls. Tuy nhien local folder chi tach `fall` va `non-fall`, khong co metadata rieng de danh dau 4 near-fall scenario. Cho baseline, nen gan tat ca file trong `non-fall` la `fall_label = 0`.

### 3.3. Sensor va cot trong `.mat`

Moi `.mat` file co du cac key:

| Key | Shape local | Y nghia |
| --- | --- | --- |
| `ax`, `ay`, `az` | `(N, 1)` | accelerometer, don vi `g`, gravity removed theo README |
| `w`, `x`, `y`, `z` | `(N, 1)` | quaternion |
| `droll`, `dpitch`, `dyaw` | `(N, 1)` | angular velocity / gyro-derived channel |
| `heart` | `(N, 1)` | PPG/heart signal |
| `time` | `(N, 6)` | real time fields |

Voi DS-Fall baseline, lay:

`[ax, ay, az, droll, dpitch, dyaw]`

Khong dung `heart` va quaternion trong baseline 6-axis. Co the dung `heart` trong bien the multimodal sau, nhung khi do khong con la DS-Fall 6-axis IMU thu gon.

### 3.4. Activity va nhan

Fall mapping:

| File | Mo ta | Direction label |
| --- | --- | --- |
| `fall1.mat` | Clockwise forward fall | `forward` |
| `fall2.mat` | Clockwise backward fall | `backward` |
| `fall3.mat` | Right-to-left lateral fall | `lateral` |
| `fall4.mat` | Counterclockwise forward fall | `forward` |
| `fall5.mat` | Counterclockwise backward fall | `backward` |
| `fall6.mat` | Left-to-right lateral fall | `lateral` |

Non-fall mapping:

`bed`, `chair`, `clap`, `cloth`, `eat`, `hair`, `shoe`, `stair`, `teeth`, `walk`, `wash`, `write`, `zip` -> `fall_label = 0`, `direction_label = none`.

### 3.5. De xuat loc cho DS-Fall

HIFD la dataset phu tot vi cung 50 Hz va cung wrist placement. Cach xu ly:

1. Load `.mat` bang `scipy.io.loadmat`.
2. Tao matrix `N x 6` theo thu tu `[ax, ay, az, droll, dpitch, dyaw]`.
3. Voi fall trial, neu khong co event timestamp, tim peak tren `sqrt(ax^2 + ay^2 + az^2)` roi cat 100 samples quanh peak.
4. Voi non-fall trial, cat sliding window 100 samples, stride 50 samples, sau do cap so window theo subject/activity.
5. Split theo `subject_id`, khong split theo window.
6. Vi fall direction khong can bang, dung class weight hoac undersample direction lon.

## 4. `Dataset` / BITS-2 Geriatric Wrist-Worn Dataset

### 4.1. Duong dan va cau truc

Path:

`data/raw/Dataset`

Cau truc local:

```text
Dataset/
  adl/
    user1/
      user1_adl1.csv
      ...
      user1_adl16.csv
  fall/
    user1/
      user1_fall1.csv
      ...
      user1_fall8.csv
```

Local copy co:

| Thanh phan | So luong |
| --- | ---: |
| CSV files | 984 |
| Users | 41 |
| ADL files | 656 = 41 x 16 |
| Fall files | 328 = 41 x 8 |
| ADL ids | `adl1` to `adl16` |
| Fall ids | `fall1` to `fall8` |

Co 4 file cua `user31` dat ten bat thuong nhung van doc duoc:

```text
fall/user31/user31_.fall1.csv
fall/user31/user31_.fall3.csv
fall/user31/user31_.fall4.csv
fall/user31/user31_.fall6.csv
```

Parser nen lay activity id bang regex linh hoat, khong nen phu thuoc duy nhat vao pattern `userXX_fallY.csv`.

### 4.2. Quan he voi BITS-2 paper

Dataset nay khop voi mo ta BITS-2 / geriatric wrist-worn dataset:

- 41 volunteers.
- 16 ADL activities.
- 8 fall activities.
- Thiet bi custom deo co tay trai.
- Motion sensors 20 Hz, heart-rate 1 Hz.
- Sensor gom accelerometer, gyroscope, magnetometer, linear accelerometer, heart-rate.

Luu y quan trong: paper goc noi moi activity lap 5 trials, tong 4920 instances. Ban local hien tai chi co 984 CSV, tuong ung 41 users x 24 activities, khong co trial id rieng trong ten file. Vi vay khi lam pipeline tren ban local, nen coi moi file la mot sequence theo `subject + activity`, khong gia dinh co 5 file trial rieng.

### 4.3. Format CSV local

Header chinh:

```text
t,x,y,z,a,
```

File tron nhieu sensor theo cot cuoi:

| Sensor label local | Y nghia ky thuat | Co nen dung cho DS-Fall baseline |
| --- | --- | --- |
| `acg` | accelerometer co thanh phan gravity, magnitude quanh 9.8 khi dung yen | Co, neu muon giu thong tin tu the/gravity |
| `acc` | kha nang cao la linear acceleration/gravity-removed acceleration | Co the dung thay `acg` neu muon align voi HIFD gravity-removed |
| `gyro` | gyroscope 3 truc | Co |
| `mgm` | magnetometer 3 truc | Khong trong baseline |
| `hrt` | heart-rate, dong co dang `t,bpm,a,hrt,,` | Khong trong baseline |

Thong ke sensor rows moi file:

| Sensor | Min rows | Median rows | Max rows |
| --- | ---: | ---: | ---: |
| `acc` | 18 | 479.0 | 5758 |
| `acg` | 19 | 480.5 | 5759 |
| `gyro` | 19 | 480.5 | 5759 |
| `mgm` | 18 | 463.5 | 5606 |
| `hrt` | 2 | 28.0 | 271 |

Co 375 file chua header lap lai ben trong file. Khi parse can bo qua cac dong co `row[0] == "t"`.

### 4.4. Van de timestamp local

Cot `t` trong ban local dang o scientific notation bi lam tron manh, vi du nhieu dong lien tiep cung `6.01E+13`. Mot so file chi con 1 den 3 timestamp unique string cho hang nghin sample. Vi vay:

- Khong nen tinh sampling rate tu cot `t` local.
- Khong nen align sensor bang timestamp local.
- Nen dung thu tu dong sau khi tach sensor.
- Sampling rate nen lay theo paper goc: motion 20 Hz, heart-rate 1 Hz.

### 4.5. Activity mapping

ADL:

| Local id | Activity |
| --- | --- |
| `adl1` | Walking slowly |
| `adl2` | Walking quickly |
| `adl3` | Jogging |
| `adl4` | Jumping |
| `adl5` | Climbing up slowly |
| `adl6` | Climbing down slowly |
| `adl7` | Climbing up normally |
| `adl8` | Climbing down normally |
| `adl9` | Slowly sitting on chair |
| `adl10` | Rapidly sitting on chair |
| `adl11` | Nearly sitting on chair and getting up |
| `adl12` | Swinging hands |
| `adl13` | Lying on bed |
| `adl14` | Lying on back and getting up slowly |
| `adl15` | Lying on back and getting up normally |
| `adl16` | Transition from sideways to back while lying |

Fall:

| Local id | Activity | Direction label de xuat |
| --- | --- | --- |
| `fall1` | Forward fall landing on knee | `forward` |
| `fall2` | Right fall | `lateral` hoac `right` neu dung 4-class |
| `fall3` | Left fall | `lateral` hoac `left` neu dung 4-class |
| `fall4` | Forward fall | `forward` |
| `fall5` | Seated on bed and falling on ground | `other`/exclude khoi direction baseline |
| `fall6` | Forward fall body weight on hand | `forward` |
| `fall7` | Backward fall from seated position | `backward` |
| `fall8` | Grabbing while falling | `other`/exclude khoi direction baseline |

Cho DS-Fall 3-class direction (`forward`, `backward`, `lateral`), nen:

- Dung `fall1`, `fall4`, `fall6` -> `forward`.
- Dung `fall7` -> `backward`.
- Dung `fall2`, `fall3` -> `lateral`.
- Dung `fall5`, `fall8` cho binary fall detection, nhung khong tinh direction loss, hoac gan `other` neu mo rong direction head.

### 4.6. De xuat loc cho DS-Fall

Co 2 lua chon accelerometer:

| Lua chon | Kenh | Khi nao dung |
| --- | --- | --- |
| Raw/gravity-aware | `[acg_x, acg_y, acg_z, gyro_x, gyro_y, gyro_z]` | Phu hop neu muon giu tu the va thay doi gravity nhu WEDA |
| Linear/gravity-removed | `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]` | Phu hop neu muon gan hon voi HIFD, vi HIFD da remove gravity |

Khuyen nghi thuc te:

1. Neu train rieng BITS-2: bat dau voi `acg + gyro`.
2. Neu train chung voi HIFD: thu ca hai bien the `acg + gyro` va `acc + gyro`, so sanh validation.
3. Khong dung `mgm` va `hrt` trong DS-Fall baseline.
4. Do 20 Hz, co hai cach tao `100 x 6`:
   - Giu 20 Hz va cat 5 giay = 100 samples.
   - Resample len 50 Hz va cat 2 giay = 100 samples.
5. Vi timestamp local khong tin cay, neu resample thi resample theo index voi gia dinh 20 Hz, khong resample theo cot `t`.
6. Neu training chung WEDA/HIFD/BITS, nen ghi metadata `sampling_rate_original` va `resampled = true/false`.

## 5. Unified Label Schema

Nen chuan hoa metadata moi window thanh:

| Field | Kieu | Y nghia |
| --- | --- | --- |
| `dataset` | string | `weda`, `hifd`, `bits2` |
| `source_path` | string | path file goc |
| `subject_id` | string | user/subject id |
| `activity_id` | string | id goc, vi du `F01`, `fall1`, `adl7` |
| `activity_name` | string | ten activity da map |
| `fall_label` | int | 0 non-fall, 1 fall |
| `direction_label` | string | `forward`, `backward`, `lateral`, `left`, `right`, `other`, `none` |
| `direction_supervised` | bool | `true` chi khi direction label ro rang |
| `sampling_rate_original` | float | 50 hoac 20 |
| `sampling_rate_model` | float | 50 neu chuan hoa ve DS-Fall 2s |
| `window_start_idx` | int | index sau khi align/resample |
| `window_end_idx` | int | index sau khi align/resample |
| `event_source` | string | `timestamp`, `peak_acc`, `sliding` |

Direction loss chi nen tinh khi:

```text
fall_label == 1 and direction_supervised == true
```

## 6. Unified Preprocessing Pipeline

### 6.1. Extract 6-axis IMU

WEDA:

```text
accel.csv + gyro.csv -> [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
```

HIFD:

```text
.mat -> [ax, ay, az, droll, dpitch, dyaw]
```

BITS-2:

```text
CSV rows where label == acg or acc
CSV rows where label == gyro
-> [ax, ay, az, gx, gy, gz]
```

### 6.2. Length va quality filters

Loai sample/window neu:

- Thieu accelerometer hoac gyroscope.
- Sau resample/cat window khong du 100 timestep va khong muon pad.
- Co NaN/Inf.
- Sensor rows qua ngan: voi 50 Hz can it nhat 100 samples, voi BITS 20 Hz can it nhat 100 samples neu giu 5 giay.
- BITS CSV chi con header/heart-rate hoac parse duoc sensor row bat thuong.

### 6.3. Unit normalization

Khong nen tron raw units giua dataset ma khong normalize:

- WEDA accelerometer nhin theo m/s^2.
- HIFD accelerometer theo `g`, gravity removed.
- BITS `acg` nhin theo m/s^2; `acc` co xu huong gravity-removed.
- Gyro units co the khac nhau giua dataset.

De xuat:

1. Chuyen acceleration ve cung don vi, vi du `m/s^2`.
   - Neu input la `g`, nhan `9.80665`.
2. Sau do standardize theo channel bang mean/std tinh tren train split.
3. Khong tinh normalization tren test subject de tranh leakage.
4. Luu lai scaler theo dataset/channel neu muon cross-dataset evaluation ro rang.

### 6.4. Windowing

| Dataset | Fall window | Non-fall window |
| --- | --- | --- |
| WEDA | Uu tien `fall_timestamps.csv`, sau do peak acceleration trong khoang timestamp | Sliding 2s, stride 1s, cap windows |
| HIFD | Peak acceleration vi khong co timestamp event local | Sliding 2s, stride 1s, cap windows |
| BITS-2 | Peak acceleration, 5s o 20 Hz hoac resampled 2s o 50 Hz | Sliding 5s neu giu 20 Hz, hoac 2s neu resample |

### 6.5. Splitting

Bat buoc split theo subject:

- WEDA: split theo `Uxx`; nen co evaluation rieng tren elderly ADL `U21-U31` de do false positive.
- HIFD: split theo `subject_xx`.
- BITS-2: split theo `userX`.

Khong split theo file/window vi cung mot trial co the sinh nhieu window gan nhau.

### 6.6. Class balancing

Khuyen nghi:

- Fall detection: giu ti le non-fall/fall khoang 1:1 den 3:1 trong train.
- Direction: balance theo `forward`, `backward`, `lateral`.
- WEDA co 350 fall timestamp rows va nhieu ADL windows, nen phai undersample/cap ADL.
- HIFD co `fall4`, `fall5`, `fall6` thieu subject, nen dung class weight.
- BITS-2 co direction `other` ambiguous, nen khong ep vao 3-class neu chua co ly do.

## 7. Dataset-Specific Recommendation

### WEDA-FALL

Nen dung lam dataset chinh vi:

- Wrist smartwatch.
- 50 Hz dung voi DS-Fall 2s x 50 Hz = 100 timestep.
- Co accelerometer + gyroscope day du.
- Co fall direction forward/backward/lateral ro.
- Co `fall_timestamps.csv` de cat dung vung actual fall.
- Co elderly ADL de test false positive tren nhom nguoi gia.

Loc de train:

```text
dataset/50Hz/{F01..F08,D01..D11}
only accel + gyro
fall window from fall_timestamps
ADL sliding windows with cap
```

### HIFD

Nen dung lam dataset phu vi:

- Cung 50 Hz.
- Left wrist.
- Co 6 fall type map tot vao 3 huong.
- Co hard non-fall/near-fall-like scenario.

Han che:

- Khong co timestamp event.
- Mot so fall type khong du subject.
- Accelerometer da remove gravity, khac WEDA.

Loc de train:

```text
fall/*.mat -> y_fall=1, direction by fall id
non-fall/*.mat -> y_fall=0, direction=none
channels = ax, ay, az, droll, dpitch, dyaw
fall windows centered at peak acceleration
```

### BITS-2 / `Dataset`

Nen dung can than vi:

- Wrist-only, 41 users, nhieu ADL/fall.
- Co accelerometer + gyroscope + heart-rate + magnetometer.
- Nhung local timestamp bi lam tron, khong nen align theo `t`.
- Sampling rate 20 Hz khac voi WEDA/HIFD.
- Local copy khong co trial id rieng du paper goc noi moi activity lap 5 trials.

Loc de train/eval:

```text
adl/*.csv -> y_fall=0
fall/*.csv -> y_fall=1
channels = acg + gyro first; acc + gyro as ablation
use row order, assume 20 Hz
fall5/fall8: binary fall only or direction=other
```

Vai tro tot nhat:

- External validation sau khi train tren WEDA/HIFD.
- Ablation ve resampling 20 Hz -> 50 Hz.
- Kiem tra mo hinh tren custom medical wrist device.

## 8. Sources Checked

Local sources:

- `data/raw/WEDA-FALL-main/README.md`
- `data/raw/WEDA-FALL-main/dataset/fall_timestamps.csv`
- `data/raw/HR_IMU_falldetection_dataset-master/README.md`
- Mau CSV va `.mat` trong 3 thu muc raw.
- `doc/DS-Fall _ Dual-Stream Depthwise-Separable Temporal Network for Direction-Sensitive Fall Detection.md`

External source used for `Dataset`/BITS-2 activity names and sampling-rate confirmation:

- Data in Brief article: "Inertial measurement and heart-rate sensor-based dataset for geriatric fall detection using custom built wrist-worn device", DOI `10.1016/j.dib.2023.109812`.
- PMC mirror: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10709028/`
