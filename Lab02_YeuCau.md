# Lab 02 – Làm quen với thuật toán Q-learning

**Môn học:** Học máy tăng cường cho các hệ thống mạng  
**GVHD:** ThS. Phan Trung Phát – phatpt@uit.edu.vn  
**Khoa:** Mạng máy tính và Truyền thông – UIT  
**Tái bản lần 1 – Tháng 03/2026**

---

## A. MỤC TIÊU

- Giới thiệu tổng quát phân loại các nhóm giải thuật học tăng cường.
- Làm quen với thuật toán Q-learning, cách hiện thực và phân tích các trường hợp ứng dụng.
- Ứng dụng Q-learning vào giải quyết các bài toán cổ điển dựa trên môi trường được cung cấp sẵn và môi trường tự xây dựng, đặc biệt là bài toán Load Balancing.
- Thực hiện đánh giá và kiểm thử trên các chính sách.

---

## B. GIỚI THIỆU

### 1. Sơ lược về phân loại các giải thuật Reinforcement Learning

#### a. Model-based và Model-free Reinforcement Learning

Trong **Học tăng cường** (Reinforcement Learning, RL), mục tiêu của một tác nhân (Agent) là học cách đưa ra quyết định tối ưu thông qua việc tương tác với môi trường (Environment). Agent quan sát trạng thái, thực hiện hành động, sau đó nhận phản hồi dưới dạng phần thưởng và trạng thái mới.

**Model-based RL:**
- Agent xây dựng hoặc đã biết trước "mô hình" của thế giới:
  - Xác suất chuyển trạng thái P(s'|s, a)
  - Hàm phần thưởng R(s, a)
- Giống như có bản đồ đầy đủ, Agent có thể lập kế hoạch trước.

**Model-free RL:**
- Agent không cố xây dựng mô hình môi trường.
- Học trực tiếp từ trải nghiệm (thử và sai).
- Phù hợp với hệ thống phức tạp như mạng máy tính.

#### b. Value-based và Policy-based methods

**Value-based methods:**
- Học cách đánh giá mức độ tốt của các trạng thái hoặc hành động thông qua hàm giá trị Q(s, a).
- Policy tối ưu: π(s) = argmax_a Q(s, a)
- Học đánh giá trước, hành động dựa trên đánh giá.

**Policy-based methods:**
- Học trực tiếp chiến lược hành động.
- Policy mô hình hóa như hàm xác suất.
- Tối ưu hóa trực tiếp tổng phần thưởng kỳ vọng.

#### c. On-policy và Off-policy

- **On-policy** (π = b): Agent học dựa trên chính policy đang sử dụng. Ổn định nhưng khó tái sử dụng dữ liệu.
- **Off-policy** (π ≠ b): Agent học policy khác với policy thu thập dữ liệu. Tái sử dụng dữ liệu hiệu quả (experience replay).

### 2. Giới thiệu về thuật toán Q-learning

#### a. Giới thiệu và ý tưởng

- Q-learning thuộc nhóm **Value-based**, **Model-free**, **Off-policy**.
- Ý tưởng: Agent học hàm giá trị **Q-function** để đánh giá chất lượng từng hành động trong từng trạng thái.
- Công thức cập nhật (Temporal Difference):

```
Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') − Q(s, a)]
```

Trong đó:
- `s`: trạng thái hiện tại
- `a`: hành động được thực hiện
- `r`: phần thưởng nhận được
- `s'`: trạng thái tiếp theo
- `α`: learning rate
- `γ`: hệ số chiết khấu

#### b. Cách hiện thực

- Dùng **Q-table** (ma trận N×M) lưu trữ Q(s, a).
- Ban đầu khởi tạo bằng 0.
- Các tham số quan trọng:
  1. **Learning rate (α)**: Mức độ ảnh hưởng của thông tin mới.
  2. **Discount factor (γ)**: Mức độ quan trọng của phần thưởng tương lai.
  3. **Exploration rate (ε)**: Cân bằng exploration vs exploitation qua **ε-greedy policy**.

---

## C. THỰC HÀNH

### 1. Hiện thực Q-learning với môi trường "Frozen Lake"

**Mô tả:** Sinh viên thực hiện tìm hiểu môi trường sẵn có "Frozen Lake" được cung cấp bởi Gymnasium để tiến hành hiện thực thuật toán Q-learning.

**Các bước:**

- **Bước 1:** Nhìn nhận bài toán. Frozen Lake là bài toán điều hướng trên lưới 4×4, đi từ S đến G, tránh H (hố nước).

- **Bước 2:** Tìm hiểu các biểu diễn MDP:
  - ❓ Observation Space mô tả điều gì?
  - ❓ Action Space mô tả điều gì?
  - ❓ Reward được định nghĩa như thế nào?
  - ❓ Nguyên tắc hoạt động của nhân vật?
  - ❓ Khi nào thì bài toán kết thúc?

- **Bước 3:** Cài đặt thư viện: `pip install -r requirements.txt`

- **Bước 4:** Mở file `Lab2.ipynb`, nạp môi trường:
  ```python
  env = gym.make("FrozenLake-v1", is_slippery=False)
  agent = QLearningAgent(env)
  ```

- **Bước 5:** Hoàn thành các block code còn thiếu (training loop, get_q_value, choose_greedy_action, ...).

- **Bước 6:** Quan sát kết quả.
  - ❓ Hãy thử suy nghĩ việc hiện thực Q-learning trên môi trường **Mountain Car** có thể gặp phải những rào cản nào? Có thể giải quyết bằng cách gì?

---

## D. YÊU CẦU & NỘP BÀI

### 1. Yêu cầu

#### Yêu cầu 1 – FrozenLake: Q-learning vs Random

Đối với FrozenLake Environment, hoàn thành đoạn mã định nghĩa Q-learning còn thiếu trong file `Lab2.ipynb`, **Phần 1**. Đồng thời, thêm 1 lựa chọn để có thể thực hiện chính sách **Random** trong hàm `choose_action` và thực hiện chạy với cùng episode với optimal policy được chọn ra từ thuật toán Q-learning. Lưu trữ lại Reward qua từng episode, vẽ biểu đồ chồng giữa chính sách dựa trên Q-learning và Random. Từ đó, bạn có đánh giá gì đối với từng loại policy ở trên, giải thích, minh chứng.

> **Gợi ý:** sử dụng thư viện Matplotlib.

#### Yêu cầu 2 – VacuumCleaner: Tích hợp Q-learning

Đối với VacuumCleaner Environment, dựa trên các đoạn mã được cung cấp tại câu 1 với FrozenLake Environment, chỉnh sửa lại môi trường để có khả năng tích hợp với Q-learning và tiến hành hiện thực Q-learning trong file `Lab2.ipynb`, **Phần 2**. Hiện thực chính sách **Round Robin** và **Priority-based** tương tự tại bài thực hành số 1 trên môi trường đã chỉnh sửa.

#### Yêu cầu 3 – VacuumCleaner: Đánh giá

Sau khi huấn luyện với Q-learning, thực hiện chạy môi trường với chính sách dựa trên Q-learning, Round Robin và Priority-based trong **100 episode**. Viết hàm để tổng hợp Reward nhận được dưới dạng file `.csv`, vẽ biểu đồ thể hiện Reward đạt được theo thời gian. Bạn có đánh giá gì trên các giải pháp đã hiện thực và kết quả đạt được.

#### Yêu cầu 4 – FrozenLake: So sánh siêu tham số

Thực hiện huấn luyện lại Q-learning với 3 trường hợp, tương ứng với 3 trọng số (Learning Rate, Discount Factor, Exploration Rate):

| Trường hợp | α (Learning Rate) | γ (Discount Factor) | ε (Exploration Rate) |
|:---:|:---:|:---:|:---:|
| **1** | 0.5 | 0.2 | 0.1 |
| **2** | 0.1 | 0.95 | 0.2 |
| **3** | 0.1 | 0.8 | 0.7 |

Hãy đánh giá về tốc độ hội tụ trong từng trường hợp, Reward đạt được. Từ đó, bạn có đánh giá gì trên 3 nhóm trọng số trên?

#### Yêu cầu 5 – LoadBalancing: Tích hợp Q-learning

Đối với LoadBalancing Environment, chỉnh sửa lại môi trường để có khả năng tích hợp với Q-learning và tiến hành hiện thực Q-learning trong file `Lab2.ipynb`, **Phần 3**. Hiện thực chính sách **Round Robin** tương tự tại bài thực hành số 1 trên môi trường đã chỉnh sửa.

#### Yêu cầu 6 – LoadBalancing: Đánh giá

Sau khi huấn luyện với Q-learning, thực hiện chạy môi trường với chính sách dựa trên Q-learning, Round Robin trong **100 episode**. Viết hàm lưu trữ các metrics vào file `.csv`, các metrics cần lưu bao gồm:

- Tổng thời gian chạy
- Tổng tác vụ đã tạo ra
- Số tác vụ được tiếp nhận để xử lý
- Số lượng tác vụ bị bỏ qua
- Số lượng tác vụ đã hoàn thành
- Phần thưởng

#### Yêu cầu 7 – LoadBalancing: Tổng hợp metrics

Từ các số liệu thu thập được, viết hàm tính để tổng hợp các metrics vào điền vào bảng so sánh:

| Metric | Q-learning | Round Robin |
|:---|:---:|:---:|
| **Average Reward** | | |
| **Drop Rate** | | |
| **Completion Rate** | | |

Vẽ biểu đồ thể thiện sự biến thiên của từng policy theo thời gian với 3 metrics: **reward**, **drop**, **completion**. Bạn có đánh giá gì đối với từng loại policy ở trên, giải thích, minh chứng.

---

### 2. Yêu cầu nộp bài

- Sinh viên tìm hiểu và thực hành theo hướng dẫn. Thực hiện **theo nhóm**.
- Sinh viên báo cáo kết quả thực hiện và nộp bài bằng file. Trong đó:
  - Trình bày chi tiết quá trình thực hành và trả lời các câu hỏi nếu có (kèm theo các ảnh chụp màn hình tương ứng).
  - Giải thích các kết quả đạt được.
  - Tải mẫu báo cáo thực hành và trình bày theo mẫu được cung cấp.

**Định dạng tên file nén:**

```
NhomY-LabX_MSSV1_MSSV2
```

> Ví dụ: `Nhom1-Lab02_29520001_29520002_29520003`

- Nộp báo cáo trên theo thời gian đã thống nhất tại website môn học.
- **Các bài nộp không tuân theo yêu cầu sẽ KHÔNG được chấm điểm.**

---

*Chúc các bạn hoàn thành tốt bài thực hành!*
