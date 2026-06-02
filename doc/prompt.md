Bạn là một senior ML engineer, research engineer và TensorFlow/Keras expert. Hãy đọc kỹ toàn bộ repo DS-Fall hiện tại, bao gồm cấu trúc thư mục, notebook, scripts, config, data processing, model definition, training loop, evaluation, logging và các file liên quan đến model DS-Fall-RD / A5WCEFW / A5 variant.

Mục tiêu là xây dựng pipeline nghiêm túc để xử lý hai dataset BITS và WEDA-FALL về cùng chuẩn 25 Hz, train/test nhiều kịch bản cross-dataset, đánh giá đầy đủ metric cho paper, và lưu lại toàn bộ artifacts.

Điểm quan trọng mới:

WEDA-FALL: KHÔNG downsample từ 50 Hz nữa.
WEDA-FALL: phải lấy trực tiếp folder/data 25Hz có sẵn trong raw dataset để process.

BITS: dữ liệu gốc khoảng 20 Hz.
BITS: nội suy nhẹ từ 20 Hz lên 25 Hz bằng time-based interpolation.

Sau xử lý, BITS và WEDA phải đồng bộ hoàn toàn:
- sampling rate
- window length
- input shape
- feature order
- unit
- label mapping
- scaler
- split logic
- metadata
- evaluation protocol
1. Mục tiêu chính

Nâng cấp repo để chạy thí nghiệm:

Dataset: BITS + WEDA-FALL
Sampling rate: 25 Hz
Window duration: 2 seconds
Input shape: 50 × 12
Model: A5WCEFW / A5 variant of DS-Fall-RD
Tasks:
1. Fall detection
2. Fall direction recognition

Tất cả data sau xử lý phải thống nhất:

sampling_rate = 25
window_seconds = 2.0
window_size = 50
n_features = 12
input_shape = (50, 12)

Tuyệt đối không dùng:

100 × 12

cho thí nghiệm 25 Hz.

Vì:

25 Hz × 2 seconds = 50 timesteps

Nếu code cũ hard-code:

window_size = 100

phải refactor thành:

window_size = int(sampling_rate * window_seconds)
2. Input feature 50 × 12

Mỗi timestep phải gồm đúng 12 feature theo thứ tự cố định:

0  ax
1  ay
2  az
3  gx
4  gy
5  gz
6  acc_mag
7  gyro_mag
8  jerk
9  roll
10 pitch
11 tilt_delta

Công thức:

dt = 1.0 / 25.0

acc_mag = sqrt(ax**2 + ay**2 + az**2)
gyro_mag = sqrt(gx**2 + gy**2 + gz**2)

jerk = gradient(acc_mag, dt)

roll = atan2(ay, az)

pitch = atan2(
    -ax,
    sqrt(ay**2 + az**2)
)

tilt = arccos(
    clip(az / max(acc_mag, eps), -1.0, 1.0)
)

tilt_delta = gradient(tilt, dt)

Yêu cầu kỹ thuật:

- Handle NaN và Inf.
- Dùng eps để tránh chia cho 0.
- Clip input của arccos vào [-1, 1].
- Không đổi thứ tự feature giữa BITS và WEDA.
- Lưu feature_order vào metadata.
- In thống kê mean/std/min/max/p95/p99 cho từng feature sau xử lý.
3. Đọc kỹ raw data BITS

Trước khi code, hãy inspect kỹ raw data BITS:

- Folder structure.
- File format.
- Sensor columns.
- Timestamp column nếu có.
- Subject ID.
- Trial ID.
- Activity name.
- Fall / non-fall label.
- Direction information nếu có.
- Sampling rate gốc.
- Unit của accelerometer.
- Unit của gyroscope.

BITS đang khoảng 20 Hz, cần nội suy nhẹ lên 25 Hz:

BITS: 20 Hz → 25 Hz

Yêu cầu resample BITS:

- Không upsample lên 50 Hz.
- Không duplicate sample.
- Không nối nhiều trial rồi mới resample.
- Resample từng trial độc lập.
- Nếu có timestamp thật, dùng timestamp thật.
- Nếu không có timestamp đáng tin cậy, tạo timestamp đều theo 20 Hz.
- Dùng time-based interpolation cho từng kênh sensor.
- Sau resample, kiểm tra số sample, duration và effective sampling rate.
- Không để leakage giữa các trial hoặc subject.

Gợi ý implement:

target_hz = 25
source_hz = 20
target_dt = 1.0 / target_hz

# Với mỗi trial:
# 1. tạo time_old
# 2. tạo time_new từ start đến end theo bước 1/25
# 3. interpolate từng cột ax, ay, az, gx, gy, gz

Không được interpolate sau khi đã ghép nhiều trial lại với nhau.

4. Đọc kỹ raw data WEDA-FALL

Điểm cực kỳ quan trọng:

WEDA-FALL đã có folder/data 25Hz trong raw dataset.
Không được lấy folder 50Hz rồi downsample xuống 25Hz nếu folder 25Hz có sẵn.
Phải tìm đúng folder 25Hz trong raw WEDA-FALL và process trực tiếp từ đó.

Trước khi code, hãy inspect kỹ raw WEDA-FALL:

- Folder structure.
- Xác định chính xác folder 25Hz.
- File format của folder 25Hz.
- Sensor columns.
- Timestamp column.
- Accelerometer columns.
- Gyroscope columns.
- Orientation / vertical acceleration nếu có.
- Fall begin/end timestamp nếu có.
- Subject ID.
- Activity code F01–F08 hoặc ADL.
- Trial ID.
- Fall / non-fall label.
- Direction label.
- Unit của accelerometer.
- Unit của gyroscope.

Yêu cầu riêng cho WEDA:

WEDA-FALL: dùng trực tiếp data 25Hz từ raw folder.
Không downsample lại từ 50Hz.
Không interpolate nếu data 25Hz đã đều và sạch.
Chỉ resample lại nếu audit cho thấy timestamp không đều hoặc effective Hz sai đáng kể.

Audit bắt buộc với WEDA 25Hz:

- In đường dẫn folder 25Hz đang dùng.
- In số file/trial đọc được.
- In duration mỗi trial hoặc sample count summary.
- Kiểm tra effective sampling rate.
- Nếu effective Hz quanh 25 Hz, giữ nguyên.
- Nếu effective Hz lệch nhiều, log rõ và chỉ khi cần mới time-align về 25 Hz.

Không được tự ý dùng folder 50Hz nếu folder 25Hz tồn tại.

5. Đồng bộ hóa đơn vị sensor

Dữ liệu cuối cùng của cả BITS và WEDA phải thống nhất:

Accelerometer: m/s²
Gyroscope: rad/s

Nếu raw data đang ở:

acc in g       → multiply by 9.80665
gyro in deg/s  → multiply by pi / 180

Không được assume bừa. Phải audit từng dataset bằng statistic:

acc_mag_median
acc_mag_p95
acc_mag_p99
gyro_mag_median
gyro_mag_p95
gyro_mag_p99

Rule kiểm tra:

- acc_mag median quanh 1 → khả năng acc đang ở g.
- acc_mag median quanh 9.8 → khả năng acc đang ở m/s².
- gyro p99 hàng trăm → khả năng gyro đang ở deg/s.
- gyro p99 khoảng vài rad/s đến vài chục rad/s → có thể đã là rad/s.

In log rõ:

Dataset: BITS
Detected acc unit:
Detected gyro unit:
Applied conversion:
acc_mag median/p95/p99:
gyro_mag median/p95/p99:

Dataset: WEDA
Raw folder used:
Detected acc unit:
Detected gyro unit:
Applied conversion:
acc_mag median/p95/p99:
gyro_mag median/p95/p99:
6. Đồng bộ hóa trục và feature

Sau khi chuẩn hóa về 6 kênh:

ax, ay, az, gx, gy, gz

mới tính thêm 6 feature phụ:

acc_mag
gyro_mag
jerk
roll
pitch
tilt_delta

Yêu cầu:

- Tính feature sau khi đã resample BITS lên 25Hz.
- Tính feature sau khi đã chọn đúng WEDA 25Hz.
- Tính feature sau khi đã chuẩn hóa unit.
- Không tính feature phụ trước rồi mới resample.
- Không để BITS và WEDA dùng công thức khác nhau.
- Không để BITS và WEDA dùng feature order khác nhau.

Input cuối cùng phải là:

FEATURE_ORDER = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "acc_mag",
    "gyro_mag",
    "jerk",
    "roll",
    "pitch",
    "tilt_delta",
]
7. Label mapping thống nhất

Tạo label thống nhất:

fall_label:
0 = non_fall
1 = fall

direction_label:
0 = forward
1 = backward
2 = lateral
-1 = not_applicable / unknown / non_fall

Direction head chỉ tính loss trên sample có:

direction_label != -1
7.1 WEDA-FALL direction mapping

Map WEDA-FALL như sau:

F01 → forward
F02 → lateral
F03 → backward
F04 → forward
F05 → backward
F06 → forward
F07 → backward
F08 → lateral

ADL:

fall_label = 0
direction_label = -1
7.2 BITS direction mapping

Đọc kỹ activity name trong BITS rồi map cẩn thận. Mapping đề xuất:

forward fall landing on knee        → forward
forward fall body weight on hand    → forward
forward fall                        → forward
backward seated fall                → backward
right fall                          → lateral
left fall                           → lateral
seated on bed and falling on ground → unknown nếu không rõ hướng
grabbing while falling              → unknown nếu không rõ hướng

Yêu cầu quan trọng:

- Không ép nhãn direction bừa cho activity không rõ hướng.
- Fall không rõ hướng vẫn giữ fall_label = 1.
- Fall không rõ hướng đặt direction_label = -1.
- In số lượng window fall/non-fall và direction-supervised theo từng dataset.
8. Windowing ở 25 Hz

Dùng cấu hình:

sampling_rate = 25
window_seconds = 2.0
window_size = 50
stride_seconds = 0.5
stride_size = int(round(stride_seconds * sampling_rate))

Vì:

0.5 × 25 = 12.5

Chọn thống nhất một cách. Khuyến nghị:

stride_size = 12

và lưu rõ vào metadata.

Yêu cầu windowing:

- Window từng trial độc lập.
- Không tạo window vượt qua biên trial.
- Không trộn hai activity trong cùng một window.
- Không trộn hai subject trong cùng một window.
- Không trộn BITS và WEDA trước khi tạo window.
- Tạo window riêng từng dataset, sau đó mới ghép ở experiment combined.

Mỗi window metadata phải có:

window_id
dataset
subject_id
trial_id
activity
fall_label
direction_label
start_time
end_time
sampling_rate
window_size
stride_size
feature_order
source_hz
target_hz
source_folder
resampling_method
unit_conversion
9. WEDA fall interval nếu có

Nếu WEDA folder 25Hz có thông tin fall begin/end:

- Dùng timestamp/time-domain để gán fall window.
- Không dùng index 50Hz cũ nếu đang đọc folder 25Hz.
- Nếu metadata begin/end theo giây, dùng trực tiếp theo giây.
- Nếu metadata begin/end theo sample index, kiểm tra index đó tương ứng với 25Hz hay 50Hz.

Gán fall window bằng một rule rõ ràng:

Option khuyến nghị:
fall window nếu center_time nằm trong actual fall interval

Hoặc:

fall window nếu overlap với fall interval >= 30% window length

Chọn một rule và lưu vào metadata/config.

Nếu WEDA không có fall interval trong folder 25Hz hoặc khó parse:

- Gán label theo trial/activity.
- Ghi rõ limitation trong processing_report.md.
10. BITS fall interval nếu không có timestamp

Với BITS, nếu mỗi trial là một activity fall hoặc ADL:

- Gán fall_label theo trial/activity.
- Direction label theo activity mapping.

Nếu không có fall timestamp/impact timestamp:

- Không giả tạo fall interval.
- Có thể tạo option lấy window quanh impact peak, nhưng phải tách rõ với mode window toàn trial.

Khuyến nghị mặc định:

Mode 1: window toàn trial, label theo activity.
Mode 2 optional: impact-centered windows cho fall trials.

Với paper chính, nên chạy mode mặc định trước để đảm bảo minh bạch.

11. Split chống leakage

Không được random split theo window nếu cùng subject/trial có thể rơi vào cả train và test.

Yêu cầu:

- Ưu tiên split theo subject.
- Nếu subject split không khả thi, split theo trial.
- Không để cùng trial xuất hiện ở cả train và test.
- Không để cùng subject xuất hiện ở cả train và test nếu đã chọn subject split.
- Lưu split manifest ra CSV.

Mỗi split manifest cần có:

window_id
dataset
subject_id
trial_id
activity
fall_label
direction_label
split

Với cross-dataset:

BITS → WEDA:
- Train/val chỉ từ BITS.
- Test chỉ từ WEDA.
- Scaler fit trên BITS train only.

WEDA → BITS:
- Train/val chỉ từ WEDA.
- Test chỉ từ BITS.
- Scaler fit trên WEDA train only.
12. Normalization / scaler

Bắt buộc fit scaler trên train only.

Không fit scaler trên toàn dataset trước split.
Không fit scaler trên test.
Không fit scaler chung trước khi chia experiment.

Scaler gợi ý:

StandardScaler

Fit theo feature dimension:

# X_train shape: (N, T, F)
scaler.fit(X_train.reshape(-1, F))

X_train_scaled = scaler.transform(X_train.reshape(-1, F)).reshape(N_train, T, F)
X_val_scaled   = scaler.transform(X_val.reshape(-1, F)).reshape(N_val, T, F)
X_test_scaled  = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)

Với từng experiment:

E1 BITS → BITS:
scaler fit trên BITS train only.

E2 WEDA → WEDA:
scaler fit trên WEDA train only.

E3 BITS + WEDA → BITS + WEDA:
scaler fit trên combined train only.

E4 BITS → WEDA:
scaler fit trên BITS train only.

E5 WEDA → BITS:
scaler fit trên WEDA train only.

Lưu scaler:

artifacts/experiments/{experiment_id}/scaler.pkl
13. Model cần dùng: A5WCEFW / A5 variant DS-Fall-RD

Hãy tìm trong repo model hoặc config tương ứng với:

A5WCEFW
A5
DS-Fall-RD
DS-Fall-RD + tilt12
task-specific attention
weighted CE for fall
weighted CE for direction

Nếu đã có model A5WCEFW trong repo:

- Dùng đúng implementation hiện có.
- Chỉ sửa input_shape sang (50, 12).
- Không làm hỏng logic model.

Nếu chưa có implementation rõ ràng, tạo model variant mới:

A5WCEFW_25Hz

Yêu cầu kiến trúc giữ tinh thần A5:

- Dual-stream acc/gyro branch.
- Raw acc/gyro input từ 6 kênh đầu.
- 6 feature phụ đưa vào auxiliary/fusion stream.
- Depthwise-separable temporal convolution.
- Residual/dilated temporal blocks nếu repo đã có DS-Fall-RD.
- Task-specific attention cho fall head và direction head.
- Weighted CE cho fall.
- Weighted CE cho direction.
- Multi-task output:
  fall_output
  direction_output

Input:

input_shape = (50, 12)

Output:

fall_output: binary hoặc 2-class softmax
direction_output: 3-class softmax

Loss:

fall loss: weighted CE / weighted binary CE
direction loss: weighted CE only for direction_label != -1

Nếu direction label = -1:

Mask direction loss bằng sample_weight = 0 cho direction head.
14. Training configuration

Dùng config gần nhất với A5WCEFW:

optimizer = "Adam"
learning_rate = 1e-3
batch_size = 64
max_epochs = 80
early_stopping_patience = 10
reduce_lr_patience = 5
restore_best_weights = True
seed = 42

Yêu cầu:

- In log từng epoch.
- Log train loss, val loss.
- Log fall accuracy.
- Log direction accuracy.
- Log learning rate sau mỗi epoch.
- Có ReduceLROnPlateau.
- Có EarlyStopping.
- Có ModelCheckpoint.
- Sau training, tính F1 bằng sklearn.
15. Các experiment bắt buộc

Tạo script hoặc notebook chạy tự động toàn bộ thí nghiệm sau.

E1: BITS → BITS
Train: BITS train split
Validation: BITS val split
Test: BITS test split
E2: WEDA → WEDA
Train: WEDA train split
Validation: WEDA val split
Test: WEDA test split
E3: BITS + WEDA → BITS + WEDA
Train: BITS train + WEDA train
Validation: BITS val + WEDA val
Test: BITS test + WEDA test

Report thêm per-dataset metrics:

- BITS test
- WEDA test
E4: BITS → WEDA
Train: BITS train
Validation: BITS val
Test: WEDA test

Yêu cầu:

- Không dùng WEDA để train.
- Không dùng WEDA để fit scaler.
- Scaler fit trên BITS train only.
E5: WEDA → BITS
Train: WEDA train
Validation: WEDA val
Test: BITS test

Yêu cầu:

- Không dùng BITS để train.
- Không dùng BITS để fit scaler.
- Scaler fit trên WEDA train only.
16. Evaluation chuẩn paper

Sau mỗi experiment, tự động evaluate:

Fall F1
Fall precision
Fall recall
Fall accuracy
Direction macro F1
Direction accuracy
Direction per-class F1
Confusion matrix fall
Confusion matrix direction

Direction metrics:

- Chỉ tính trên samples true_direction != -1.
- Không tính ADL/non-fall vào direction macro F1.
- Report direction_n_supervised.

Lưu:

metrics.json
metrics.csv
classification_report_fall.txt
classification_report_direction.txt
confusion_matrix_fall.csv
confusion_matrix_direction.csv
per_dataset_metrics.csv
predictions.csv
training_history.csv

predictions.csv cần có:

window_id
dataset
subject_id
trial_id
activity
true_fall
pred_fall
fall_prob
true_direction
pred_direction
direction_confidence
split
experiment_id
17. Summary table cuối cùng

Sau khi chạy xong 5 experiments, tạo bảng tổng hợp:

experiment_id
train_dataset
test_dataset
sampling_rate
input_shape
n_train
n_val
n_test
fall_f1
fall_precision
fall_recall
fall_accuracy
direction_macro_f1
direction_accuracy
direction_n_supervised
best_epoch
best_val_loss
model_params
model_size_float32_kb
model_size_int8_kb_if_available

Lưu thành:

artifacts/experiments/summary_25hz_a5wcefw.csv
artifacts/experiments/summary_25hz_a5wcefw.md

In bảng này ở cuối notebook/script.

18. Validation bắt buộc trước khi train

Trước khi train, phải in audit report:

Dataset audit:
- Dataset name.
- Source folder.
- Original sampling rate.
- Target sampling rate.
- Resampling method.
- Number of subjects.
- Number of trials.
- Number of raw samples.
- Effective sampling rate before processing.
- Effective sampling rate after processing.
- Number of windows.
- Fall windows.
- Non-fall windows.
- Direction-supervised windows.
- Direction class distribution.
- Feature shape.
- NaN count.
- Inf count.

Nếu có lỗi nghiêm trọng thì dừng:

- Missing required acc/gyro columns.
- Cannot locate WEDA 25Hz folder.
- Cannot parse BITS activity labels.
- No fall samples.
- No direction-supervised samples.
- Shape mismatch.
- NaN/Inf after preprocessing.