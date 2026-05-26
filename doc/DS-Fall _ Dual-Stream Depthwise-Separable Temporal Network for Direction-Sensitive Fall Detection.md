# **DS-Fall | Dual-Stream Depthwise-Separable Temporal Network for Direction-Sensitive Fall Detection**

**DS-Fall** là một kiến trúc deep learning nhẹ cho cảm biến IMU đeo cổ tay, dùng đầu vào 6 trục gồm accelerometer và gyroscope để thực hiện hai nhiệm vụ: phát hiện ngã và phân loại hướng ngã. Mô hình được thiết kế theo hướng edge AI, ưu tiên số tham số thấp, suy luận nhanh và phù hợp triển khai trên thiết bị wearable.

## **Input Representation**

Đầu vào của mô hình là một cửa sổ tín hiệu IMU:

X∈R100×6 

Trong đó 100 timestep tương ứng khoảng 2 giây dữ liệu ở tần số lấy mẫu 50 Hz. Sáu kênh đầu vào gồm:

\[ax​,ay​,az​,gx​,gy​,gz\] 

Accelerometer cung cấp thông tin về gia tốc, va chạm, thay đổi tư thế và trạng thái sau ngã. Gyroscope cung cấp thông tin về chuyển động xoay, giúp phân biệt các hướng ngã như forward, backward và lateral.

## **Dual-Stream Sensor Encoding**

DS-Fall  tách tín hiệu đầu vào thành hai nhánh riêng:

Xacc∈R100×3   
Xgyro∈R100×3 

Nhánh accelerometer học các đặc trưng liên quan đến rơi, impact và thay đổi trọng lực. Nhánh gyroscope học các đặc trưng liên quan đến chuyển động xoay của cổ tay và thân người. Cách tách này giúp mô hình học đặc trưng theo từng loại cảm biến, thay vì trộn toàn bộ 6 kênh ngay từ đầu.

Mỗi nhánh gồm:

**Conv1D(16, k=5)**  
**DSConv1D(24, k=5)**  
**DSConv1D(32, k=3)**

Trong đó DSConv1D là depthwise-separable convolution, giúp giảm số tham số và chi phí tính toán so với convolution thông thường, phù hợp cho edge deployment.

## **Feature Fusion**

Sau hai nhánh encoder, đặc trưng được nối lại:

Facc∈R100×32   
Fgyro∈R100×32   
F= Concat( Facc,Fgyro) ∈ R100×64

Sau đó, một pointwise convolution được dùng để trộn thông tin giữa accelerometer và gyroscope:

**Pointwise Conv1D(64, k=1)**

Fusion layer giúp mô hình kết hợp tín hiệu va chạm từ accelerometer với tín hiệu xoay từ gyroscope, từ đó hỗ trợ cả phát hiện ngã và phân loại hướng ngã.

## **Temporal Encoder**

Phần temporal encoder sử dụng các khối depthwise-separable temporal convolution với dilation tăng dần:

**DS-TCN Block, channels=64, dilation=1**  
**DS-TCN Block, channels=64, dilation=2**  
**DS-TCN Block, channels=96, dilation=4**

Các khối DS-TCN giúp mô hình học được động học của cú ngã theo thời gian, từ chuyển động trước ngã, mất thăng bằng, impact đến trạng thái sau ngã. Dilation giúp mở rộng receptive field mà không làm tăng mạnh số tham số, phù hợp với tín hiệu chuỗi ngắn của IMU.

## **Gated Attention Pooling**

Thay vì dùng global average pooling, DS-Fall sử dụng gated attention pooling để tập trung vào các timestep quan trọng trong cửa sổ tín hiệu:

z=t=1T​αtht​ 

Trong đó αt\\alpha\_tαt​ là trọng số attention tại timestep ttt. Cơ chế này giúp mô hình nhấn mạnh các đoạn có thông tin quan trọng như pre-impact, impact hoặc post-fall, đồng thời giảm ảnh hưởng của nhiễu do chuyển động tay thông thường.

## **Shared Embedding and Output Heads**

Vector sau attention được đưa qua shared embedding:

**Dense(64)**  
**Dropout(0.2)**

Từ embedding chung này, mô hình tách thành hai output head:

**Fall Head:**  
**Dense(32)**  
**Dense(2), softmax**

**Direction Head:**  
**Dense(32)**  
**Dense(3 or 4), softmax**

Fall Head dự đoán hai lớp:

* non-fall  
* fall

Direction Head dự đoán hướng ngã. Với thiết lập chính, hướng ngã gồm ba lớp:

* forward  
* backward  
* lateral

## **Training Objective**

Mô hình được huấn luyện bằng loss đa nhiệm:

L=λf​Lfall​+λd​⋅1(yfall​=1)Ldir​ 

Trong đó:

Lfall​=CE( yfall​,yfall​​ )   
Ldir​=CE( ydir​,ydir​​ ) 

Direction loss chỉ được tính cho các mẫu có nhãn fall, vì mẫu ADL hoặc non-fall không có hướng ngã. Cấu trúc loss này giúp mô hình học đồng thời khả năng phát hiện ngã và hiểu hướng chuyển động của cú ngã.

## **Key Advantages**

| Component | Technical Role |
| ----- | ----- |
| Dual-stream encoder | Tách vai trò accelerometer và gyroscope |
| Depthwise-separable convolution | Giảm tham số và MACs |
| Dilated TCN | Học động học cú ngã theo thời gian |
| Gated attention pooling | Tập trung vào timestep quan trọng |
| Shared embedding | Học biểu diễn chung cho hai nhiệm vụ |
| Dual output heads | Dự đoán fall và direction trong cùng một mô hình |
| INT8-ready design | Phù hợp triển khai edge AI/TinyML |

## **Paper Description**

**DS-Fall** is a lightweight dual-stream temporal network designed for wrist-worn 6-axis IMU signals. The model separates accelerometer and gyroscope channels into modality-specific encoders to learn impact-related and rotation-related motion patterns. The extracted features are fused and processed by dilated depthwise-separable temporal convolution blocks to capture fall dynamics over a longer temporal range. A gated attention pooling layer emphasizes informative segments within the input window. Finally, two task-specific heads jointly estimate fall occurrence and fall direction using a compact shared representation.

