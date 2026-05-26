# **Dataset Structure for DS-Fall Training**

DS-Fall cần dữ liệu IMU dạng chuỗi thời gian từ thiết bị đeo cổ tay, ưu tiên 6 kênh chính:

X=\[ax​,ay​,az​,gx​,gy​,gz​\] 

Trong đó accelerometer mô tả gia tốc, va chạm, thay đổi tư thế; gyroscope mô tả chuyển động xoay, giúp phân biệt hướng ngã. Bốn dataset được chọn là WEDA-FALL, HIFD / HR\_IMU, BITS-2 và UMAFall.

## **1\. WEDA-FALL**

**WEDA-FALL** là dataset fall detection thu bằng smartwatch Fitbit Sense đeo ở cổ tay. Dữ liệu được ghi ở tần số 50 Hz và có thêm các phiên bản đã giảm tần số như 40 Hz, 25 Hz, 10 Hz và 5 Hz. Dataset có thư mục theo sampling rate, ví dụ `50Hz`, `40Hz`, `25Hz`, `10Hz`, `5Hz`, cùng file `fall_timestamps.csv` để đánh dấu vùng xảy ra ngã.

### **Sensor và kênh dữ liệu**

Dataset sử dụng dữ liệu từ:

* 3-axis accelerometer  
* 3-axis gyroscope  
* orientation sensor

Với DS-Fall, có thể ưu tiên lấy 6 kênh IMU:

\[ax​,ay​,az​,gx​,gy​,gz​\]

Orientation có thể bỏ qua nếu muốn giữ mô hình đúng hướng 6-axis IMU, hoặc dùng làm dữ liệu phụ trong phân tích nâng cao.

### **Nhãn hoạt động**

WEDA-FALL gồm hai nhóm chính:

**Fall activities:**

* **F01:** forward fall while walking caused by slip  
* **F02:** lateral fall while walking caused by slip  
* **F03:** backward fall while walking caused by slip  
* **F04:** forward fall while walking caused by trip  
* **F05:** backward fall when trying to sit down  
* **F06:** forward fall while sitting  
* **F07:** backward fall while sitting  
* **F08:** lateral fall while sitting

**ADL activities:**

* walking  
* jogging  
* stairs  
* sitting and standing  
* collapse into chair  
* crouching  
* stumble  
* jump  
* hit table with hand  
* clapping  
* opening and closing door

Các fall type này có thể map thành direction label: **forward, backward, lateral**

WEDA-FALL đặc biệt phù hợp với DS-Fall vì có dữ liệu wrist-based, sampling rate 50 Hz, có accelerometer \+ gyroscope, và có các hướng ngã forward, backward, lateral rõ ràng. Ngoài ra, paper gốc có nhắc đến các pha của cú ngã như pre-fall, impact, adjustment và post-fall, rất phù hợp với mô hình temporal network.

### **Cách dùng cho DS-Fall**

Mỗi trial được cắt thành cửa sổ:

X∈R100×6 

với 100 timestep tương ứng 2 giây ở 50 Hz. Với fall sample, nên center window quanh vùng impact hoặc vùng actual fall từ `fall_timestamps.csv`. Với ADL sample, có thể lấy sliding window 2 giây với overlap 50%.

## **2\. HIFD / HR\_IMU Fall Detection Dataset**

**HIFD**, hay **HR\_IMU Fall Detection Dataset**, là dataset thu từ thiết bị đeo ở cổ tay trái. Dataset có 21 subjects, sampling rate 50 Hz, gồm 19 scenarios: 6 falls, 9 ADLs và 4 near-falls.

### **Sensor và kênh dữ liệu**

Các trường dữ liệu chính gồm:

ax​,ay​,az​

cho accelerometer, cùng các thông tin gyroscope / orientation:

w,x,y,z​

là quaternion, và:

droll,dpitch,dyaw ​

là angular velocity của gyroscope. Ngoài ra còn có `heart` từ PPG sensor và `time` là thời gian thực.

Với DS-Fall, có thể dùng trực tiếp:

\[ax​,ay​,az​,droll,dpitch,dyaw\]

Heart rate và quaternion có thể bỏ qua trong baseline để giữ mô hình nhẹ, hoặc dùng trong phiên bản mở rộng.

### **Nhãn hoạt động**

Fall scenarios gồm:

* fall1: clockwise forward fall  
* fall2: clockwise backward fall  
* fall3: right-to-left lateral fall  
* fall4: counterclockwise forward fall  
* fall5: counterclockwise backward fall  
* fall6: left-to-right lateral fall

ADL scenarios gồm lying on bed, sitting on chair, hitting sensor, wearing clothes, eating, brushing hair, tying shoelace, stairs, brushing teeth, walking, washing, writing, zipping. Dataset cũng có near-fall, rất hữu ích để kiểm tra false positive.

### **Cách map nhãn cho DS-Fall**

Fall detection label:

yfall= 1  |  fall1–fall6   
yfall= 0  |  ADL hoặc nearfall

Direction label:

ydir​ ∈ {forward, backward, lateral} 

Mapping:

* fall1, fall4 → forward  
* fall2, fall5 → backward  
* fall3, fall6 → lateral

Near-fall nên được giữ ở class non-fall để mô hình học phân biệt fall thật với chuyển động gần giống ngã.

### **Cách dùng cho DS-Fall**

HIFD rất phù hợp làm dataset phụ vì cùng sampling rate 50 Hz với input thiết kế của DS-Fall. Mỗi trial có thể cắt thành:

X∈R100×6 

Nếu dữ liệu dài hơn 2 giây, dùng sliding window hoặc center quanh vùng peak acceleration.

## **3\. BITS-2 / Geriatric Wrist-Worn Dataset**

**BITS-2** là dataset fall detection thu bằng thiết bị custom đeo cổ tay, gồm 41 volunteers, 16 ADLs và 8 fall activities, mỗi hoạt động lặp lại 5 lần. Dữ liệu motion sensor được lấy mẫu ở 20 Hz, còn heart-rate sensor ở 1 Hz. Sensor gồm triaxial accelerometer, triaxial gyroscope, triaxial magnetometer, linear accelerometer và heart-rate sensor.

### **Sensor và kênh dữ liệu**

Data format gồm raw time-series với timestamp:

* 3-axis accelerometer  
* 3-axis gyroscope  
* 3-axis magnetometer
* linear accelerometer  
* heart rate

Với DS-Fall baseline, nên lấy:

\[ax​,ay​,az​,gx​,gy​,gz​\]

Magnetometer, linear acceleration và heart rate có thể bỏ qua trong mô hình nhẹ đầu tiên, hoặc dùng để thử nghiệm phiên bản multimodal sau.

### **Nhãn hoạt động**

Dataset có:

* 16 ADL activities  
* 8 fall activities  
* 5 trials cho mỗi activity

Do có nhiều fall type hơn, BITS-2 phù hợp nếu muốn mở rộng direction head từ 3 hướng thành 4 hướng: **forward, backward, left, right**

hoặc gom lại thành 3 hướng: **forward, backward, lateral** (trong đó left và right được gộp thành lateral)

### **Khác biệt quan trọng so với DS-Fall input**

BITS-2 có sampling rate 20 Hz, trong khi DS-Fall đang thiết kế input: 100×6

ứng với 2 giây ở 50 Hz. Vì vậy có hai cách xử lý:

**Cách 1: Giữ nguyên 20 Hz**

Dùng cửa sổ 5 giây: 5s×20Hz=100 timestep

Khi đó input vẫn là:

X∈R100×6 

nhưng mỗi sample đại diện cho 5 giây thay vì 2 giây.

**Cách 2: Resample lên 50 Hz**

Nội suy dữ liệu từ 20 Hz lên 50 Hz, sau đó cắt cửa sổ 2 giây: 2s×50Hz=100 timestep

Cách này giúp thống nhất với WEDA-FALL và HIFD, nhưng có thể tạo thêm dữ liệu nội suy không thật.

### **Cách dùng cho DS-Fall**

BITS-2 phù hợp để kiểm tra khả năng tổng quát hóa trên thiết bị đeo cổ tay custom và dữ liệu có nhiều ADL / fall hơn. Dataset này đặc biệt hữu ích nếu muốn chứng minh mô hình không chỉ hoạt động trên smartwatch mà còn có thể dùng cho wearable medical device.

## **4\. UMAFall**

**UMAFall** là dataset multisensor cho fall detection, gồm dữ liệu từ 19 subjects thực hiện các ADL và fall được định nghĩa trước. Điểm mạnh của UMAFall là có nhiều vị trí sensor trên cơ thể, giúp đánh giá ảnh hưởng của sensor placement.

### **Sensor và vị trí đeo**

UMAFall sử dụng nhiều sensing point trên cơ thể. Các nguồn mô tả dataset cho biết dữ liệu gồm accelerometer, gyroscope và magnetometer, với các vị trí như wrist, waist, ankle, chest và pocket / hip tùy thiết bị.

Với DS-Fall, chỉ nên lọc dữ liệu wrist sensor để giữ đúng bài toán wrist-worn fall detection.

### **Sensor và kênh dữ liệu**

Tùy file / thiết bị, có thể có:

* accelerometer  
* gyroscope  
* magnetometer

Với mô hình DS-Fall, nên lấy 6 kênh:

\[ax​,ay​,az​,gx​,gy​,gz​\]

Nếu một file chỉ có accelerometer hoặc thiếu gyroscope, không nên đưa vào tập train chính của DS-Fall. Có thể dùng riêng cho ablation study dạng accelerometer-only.

### **Nhãn hoạt động**

UMAFall thường được dùng cho bài toán phân biệt: fall vs. ADLfall 

Các mô tả công khai cho biết dataset có 8 loại ADL và 3 loại falls: backward, forward và lateral. Đây là nhãn direction rất phù hợp với DS-Fall.

Direction mapping:

* forward fall → forward  
* backward fall → backward  
* lateral fall → lateral

### **Khác biệt cần chú ý**

UMAFall là multisensor dataset, nên cần lọc đúng sensor ở cổ tay. Nếu trộn nhiều vị trí như ankle, chest, waist vào cùng một mô hình wrist-based, mô hình có thể học sai phân bố chuyển động. Ngoài ra, một số nghiên cứu dùng UMAFall ở 20 Hz và cắt cửa sổ 5 giây, nên khi dùng chung với WEDA-FALL / HIFD cần chuẩn hóa sampling rate hoặc chuẩn hóa số timestep.

### **Lọc data cho DS-Fall**

Pipeline đề xuất:

1. Lọc các file có sensor ở wrist.  
2. Lấy 6 kênh accelerometer \+ gyroscope.  
3. Chuẩn hóa đơn vị acceleration và angular velocity.  
4. Cắt cửa sổ quanh fall event hoặc peak acceleration.  
5. Map nhãn thành fall / non-fall và forward / backward / lateral.

UMAFall nên dùng cho cross-dataset benchmark hơn là dataset chính, vì nó không chỉ tập trung vào wrist mà có nhiều vị trí sensor khác nhau.

# **Unified Data Format for DS-Fall**

Để train DS-Fall trên nhiều dataset, nên chuẩn hóa tất cả về format chung:

Xi∈R100×6 

với:

Xi=\[ax​,ay​,az​,gx​,gy​,gz​\] 

Mỗi sample nên có metadata:

| Field | Ý nghĩa |
| ----- | ----- |
| **`subject_id`** | Mã người tham gia |
| **`dataset`** | WEDA-FALL / HIFD / BITS-2 / UMAFall |
| **`activity_id`** | Mã hoạt động gốc |
| **`fall_label`** | 0 \= non-fall, 1 \= fall |
| **`direction_label`** | forward / backward / lateral / none |
| **`sampling_rate`** | 20 Hz hoặc 50 Hz |
| **`sensor_position`** | wrist |
| **`window_start`** | timestep bắt đầu |
| **`window_end`** | timestep kết thúc |

