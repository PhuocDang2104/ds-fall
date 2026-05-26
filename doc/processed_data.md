# DS-Fall Processed Data Report

## 1. Tổng quan dữ liệu sau xử lý

Bộ dữ liệu DS-Fall sau xử lý được chuẩn hóa từ 3 nguồn dữ liệu IMU:

- BITS
- HIFD
- WEDA

Mục tiêu của bước xử lý là đưa toàn bộ dữ liệu về cùng một định dạng đầu vào cho mô hình phát hiện ngã và phân loại hướng ngã.

| Hạng mục | Giá trị |
|---|---:|
| Tổng số window | 6765 |
| Shape dữ liệu | `(6765, 100, 6)` |
| Shape mỗi sample | `(100, 6)` |
| Tần số lấy mẫu chuẩn | 50 Hz |
| Thời lượng mỗi window | 2 giây |
| Số timestep mỗi window | 100 |
| Số kênh IMU | 6 |
| Thứ tự kênh | `[ax, ay, az, gx, gy, gz]` |
| NaN | 0 |
| Inf | 0 |

Dữ liệu sau xử lý đã sạch, không có giá trị lỗi NaN hoặc Inf, và phù hợp để đưa trực tiếp vào mô hình deep learning dạng temporal như 1D-CNN, TCN hoặc Depthwise-Separable CNN.

---

## 2. Cấu trúc input của mô hình

Mỗi mẫu dữ liệu có dạng:

X ∈ R^(100 × 6)

Trong đó:

- `100` là số timestep trong 2 giây.
- `6` là số kênh cảm biến IMU.
- Các kênh gồm: `[ax, ay, az, gx, gy, gz]`

Ý nghĩa:

| Nhóm cảm biến | Kênh | Vai trò |
|---|---|---|
| Accelerometer | ax, ay, az | Ghi nhận gia tốc, va chạm, thay đổi tư thế |
| Gyroscope | gx, gy, gz | Ghi nhận chuyển động xoay, hỗ trợ phân biệt hướng ngã |

---

## 3. Thống kê theo từng dataset

| Dataset | Windows | Subjects | Fall Windows | Non-Fall Windows | Fall Ratio |
|---|---:|---:|---:|---:|---:|
| BITS | 3557 | 41 | 325 | 3232 | 9.14% |
| HIFD | 1329 | 21 | 104 | 1225 | 7.83% |
| WEDA | 1879 | 25 | 350 | 1529 | 18.63% |

Nhận xét:

- BITS là dataset lớn nhất, chiếm 3557 windows.
- WEDA có tỷ lệ fall cao nhất, khoảng 18.63%.
- HIFD có số lượng windows nhỏ nhất và tỷ lệ fall thấp nhất.
- Dữ liệu bị lệch lớp rõ rệt: số lượng non-fall lớn hơn fall rất nhiều.

Tổng quan:

| Nhóm | Số lượng |
|---|---:|
| Total windows | 6765 |
| Total fall windows | 779 |
| Total non-fall windows | 5986 |

Tỷ lệ fall toàn bộ dataset:

779 / 6765 ≈ 11.52%

Điều này cho thấy bài toán binary fall detection có mất cân bằng lớp, cần cân nhắc class weighting, focal loss hoặc balanced sampling khi train.

---

## 4. Phân bố fall và non-fall

Biểu đồ `Windows by fall label` cho thấy:

| Label | Ý nghĩa | Số lượng |
|---|---|---:|
| 0 | Non-fall | 5986 |
| 1 | Fall | 779 |

Dữ liệu non-fall chiếm phần lớn. Đây là đặc điểm thực tế của bài toán fall detection, vì trong đời sống, hoạt động bình thường xuất hiện nhiều hơn sự kiện ngã.

Tuy nhiên, nếu train trực tiếp mà không xử lý imbalance, mô hình có thể dễ thiên về dự đoán non-fall. Vì vậy khi train nên theo dõi thêm:

- Recall của lớp fall
- Precision của lớp fall
- F1-score
- Confusion matrix
- Sensitivity / specificity

Không nên chỉ nhìn accuracy.

---

## 5. Phân bố nhãn hướng ngã

| Direction Label | Supervised | Windows |
|---|---|---:|
| none | False | 5986 |
| forward | True | 280 |
| lateral | True | 217 |
| backward | True | 201 |
| other | False | 81 |

Nhận xét:

- `none` tương ứng với non-fall, không có nhãn hướng ngã.
- Các nhãn `forward`, `backward`, `lateral` là các mẫu fall có supervision hướng ngã rõ ràng.
- `other` không được dùng làm direction-supervised chính.
- Tổng số mẫu có nhãn hướng ngã rõ ràng là 698 windows.

Cách tính:

forward + backward + lateral = 280 + 201 + 217 = 698

Điều này phù hợp với thiết kế multi-task:

- Task 1: phát hiện fall / non-fall.
- Task 2: phân loại hướng ngã, chỉ áp dụng cho các mẫu fall có direction label rõ ràng.

---

## 6. Train / Validation / Test split

| Dataset | Train | Validation | Test | Subjects |
|---|---:|---:|---:|---:|
| BITS | 2501 | 528 | 528 | 41 |
| HIFD | 959 | 195 | 175 | 21 |
| WEDA | 1488 | 219 | 172 | 25 |

Tổng theo split:

| Split | Windows |
|---|---:|
| Train | 4948 |
| Validation | 942 |
| Test | 875 |

Nhận xét:

- Tập train chiếm phần lớn dữ liệu.
- Validation và test có kích thước tương đối cân bằng.
- Việc split theo subject là quan trọng để tránh subject leakage, tức là cùng một người không nên xuất hiện đồng thời trong train và test.
- Nếu split theo subject đã được đảm bảo, kết quả test sẽ phản ánh khả năng tổng quát hóa tốt hơn trên người dùng mới.

---

## 7. BITS resampling từ 20 Hz lên 50 Hz

Riêng dataset BITS được resample từ 20 Hz lên 50 Hz để đồng bộ với HIFD, WEDA và input model.

| Metric | Original Length | Resampled Length |
|---|---:|---:|
| Count | 3557 | 3557 |
| Mean | 630.43 | 1574.33 |
| Std | 598.74 | 1496.86 |
| Min | 42 | 103 |
| 25% | 277 | 691 |
| Median | 474 | 1183 |
| 75% | 780 | 1948 |
| Max | 5759 | 14396 |

Vì:

50 / 20 = 2.5

nên sau resampling, độ dài chuỗi tăng xấp xỉ 2.5 lần.

Ví dụ:

474 × 2.5 ≈ 1185

Kết quả thực tế là 1183, rất gần với giá trị kỳ vọng. Điều này cho thấy bước resampling đang hợp lý.

Ý nghĩa:

- Không tạo thêm cú ngã mới.
- Không đổi nhãn dữ liệu.
- Không đảo trật tự tín hiệu.
- Chỉ nội suy thêm các điểm trung gian để tín hiệu BITS có cùng độ phân giải thời gian 50 Hz.

Sau đó dữ liệu mới được cắt thành window cố định:

2 seconds × 50 Hz = 100 timesteps

---

## 8. Thuật toán nội suy khi resample BITS

BITS ban đầu có tần số lấy mẫu khoảng 20 Hz. Để chuyển lên 50 Hz, notebook sử dụng hướng xử lý dạng nội suy tuyến tính theo thứ tự dòng dữ liệu.

Ý tưởng chính:

- Giả định các dòng dữ liệu BITS cách đều nhau theo thời gian.
- Tạo trục thời gian gốc theo 20 Hz.
- Tạo trục thời gian mới theo 50 Hz.
- Nội suy riêng từng kênh IMU từ trục thời gian cũ sang trục thời gian mới.

Khoảng cách thời gian ban đầu:

Δt_old = 1 / 20 = 0.05 giây

Khoảng cách thời gian sau resampling:

Δt_new = 1 / 50 = 0.02 giây

Công thức nội suy tuyến tính giữa hai điểm:

x(t) = x1 + ((t - t1) / (t2 - t1)) × (x2 - x1)

Trong đó:

- `x1`, `x2` là hai giá trị tín hiệu gốc liền kề.
- `t1`, `t2` là thời điểm của hai điểm gốc.
- `t` là thời điểm mới cần ước lượng giá trị.
- `x(t)` là giá trị tín hiệu sau nội suy tại thời điểm mới.

Ví dụ:

Nếu có hai điểm gốc:

| Thời điểm | Giá trị |
|---|---:|
| 0.00 s | 1.0 |
| 0.05 s | 2.0 |

Cần tính giá trị tại 0.02 s:

x(0.02) = 1.0 + ((0.02 - 0.00) / (0.05 - 0.00)) × (2.0 - 1.0)

x(0.02) = 1.4

Nói ngắn gọn, thuật toán nối các điểm gốc bằng các đoạn thẳng, sau đó lấy thêm các điểm nằm trên các đoạn thẳng đó để đạt mật độ 50 điểm mỗi giây.

---

## 9. Chuẩn hóa dữ liệu

Trước normalization, các kênh có mean và standard deviation rất khác nhau.

Đặc biệt:

- Accelerometer và gyroscope có scale khác nhau.
- Một số kênh gyro có độ lệch chuẩn lớn hơn nhiều.
- Nếu train trực tiếp, mô hình có thể bị chi phối bởi các kênh có biên độ lớn.

Sau normalization:

- Mean của các kênh gần 0.
- Standard deviation của các kênh gần 1.

Công thức chuẩn hóa:

x_norm = (x - μ) / σ

Ý nghĩa:

- Giúp các kênh IMU có cùng scale.
- Giúp quá trình training ổn định hơn.
- Giúp optimizer hội tụ tốt hơn.
- Tránh việc một kênh có biên độ lớn lấn át các kênh khác.

Kết quả biểu đồ sau normalization cho thấy dữ liệu đã được chuẩn hóa đúng kỳ vọng.

---

## 10. Kiểm tra trực quan tín hiệu window

Các figure mẫu gồm:

- processed_forward_fall
- processed_backward_fall
- processed_lateral_fall
- processed_non_fall

### Fall windows

Các window fall có đặc điểm rõ:

- Accelerometer xuất hiện spike lớn.
- Có biến động mạnh quanh thời điểm va chạm.
- Sau va chạm, tín hiệu có xu hướng ổn định lại.
- Gyroscope cũng xuất hiện dao động, thể hiện chuyển động xoay cơ thể.

Các mẫu forward, backward và lateral có pattern khác nhau ở các trục `ax`, `ay`, `az`, giúp mô hình có thể học đặc trưng hướng ngã.

### Non-fall window

Mẫu non-fall có đặc điểm:

- Tín hiệu mượt hơn.
- Biên độ nhỏ hơn.
- Không có spike va chạm mạnh.
- Gyroscope dao động nhẹ, không có thay đổi đột ngột lớn.

Điều này xác nhận dữ liệu sau xử lý vẫn giữ được khác biệt động học giữa fall và non-fall.

---

## 11. Đánh giá chất lượng dữ liệu

| Check | Kết quả |
|---|---|
| Shape đúng chuẩn | Đạt |
| Không có NaN | Đạt |
| Không có Inf | Đạt |
| Sampling rate thống nhất | Đạt |
| Window duration thống nhất | Đạt |
| Channel order thống nhất | Đạt |
| Normalization hợp lý | Đạt |
| Fall / non-fall label rõ ràng | Đạt |
| Direction label có supervision mask | Đạt |

Dataset sau xử lý đạt yêu cầu để train mô hình DS-Fall.

---

## 12. Lưu ý khi train model

Do dữ liệu mất cân bằng lớp, khi train nên cân nhắc:

- class weighting
- balanced sampler
- focal loss
- theo dõi recall của lớp fall

Với direction classification, chỉ nên tính loss hướng ngã trên các sample có:

direction_supervised = True

Không nên ép mô hình học direction từ các mẫu `none` hoặc `other`, vì các nhãn này không đại diện cho hướng ngã chính.

Gợi ý multi-task loss:

L = L_fall + λ × M_dir × L_direction

Trong đó:

- `L_fall`: loss phát hiện fall / non-fall.
- `L_direction`: loss phân loại hướng ngã.
- `M_dir`: mask chỉ bật với mẫu có direction supervision.
- `λ`: trọng số cân bằng giữa hai task.

---

## 13. Kết luận

Bộ dữ liệu DS-Fall sau xử lý đã được chuẩn hóa thành định dạng thống nhất:

X shape = (6765, 100, 6)

Dữ liệu có 3 nguồn chính gồm BITS, HIFD và WEDA, được đồng bộ về 50 Hz, cắt thành window 2 giây và chuẩn hóa theo từng kênh. Kết quả kiểm tra cho thấy dữ liệu sạch, không có NaN/Inf, channel order nhất quán và phân bố tín hiệu hợp lý giữa fall và non-fall.

Dataset này phù hợp để huấn luyện mô hình fall detection nhẹ cho wearable edge AI, đồng thời hỗ trợ mở rộng sang direction-sensitive fall classification với ba hướng chính:

- forward
- backward
- lateral

Tuy nhiên, do dữ liệu mất cân bằng giữa non-fall và fall, quá trình huấn luyện cần đặc biệt chú ý đến recall của lớp fall và sử dụng cơ chế direction supervision mask cho nhánh phân loại hướng ngã.