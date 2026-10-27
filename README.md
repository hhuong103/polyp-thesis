# Polyp thesis

Pipeline so sánh U-Net và U-Net++ với ResNet34, augmentation đồng nhất, split theo nhóm ảnh gần trùng và nhiều seed. API vẫn hỗ trợ checkpoint E3/E4 cũ để đối chiếu, nhưng ghi rõ giới hạn dữ liệu và nguồn gốc huấn luyện.

## Cài đặt

Môi trường đã kiểm tra: Python 3.10 trên Windows; môi trường Linux riêng dùng Python 3.12. Dùng virtualenv riêng cho mỗi hệ điều hành, không dùng `.venv/Scripts/python.exe` như Python Linux.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/cpu.txt
python -m pip install -r requirements.txt
python -m pip check
```

`requirements/base.txt` ghim dependency chính; `requirements/train.txt` có matplotlib cho trực quan hóa; `backend/requirements.txt` có dependency API. Các bản `lock-*.txt`, khi có, ghi toàn bộ dependency của môi trường kiểm chứng tương ứng. Bản CPU dùng index PyTorch chính thức. Không áp dụng bản lock Windows lên Linux hoặc ngược lại. Nếu dùng GPU, cài đúng cặp torch/torchvision tương ứng với CUDA từ hướng dẫn PyTorch và ghi lại môi trường; không kỳ vọng resume bitwise qua các môi trường khác nhau.

## Dữ liệu và split

Đặt 1.000 cặp Kvasir-SEG tại `data/raw/kvasir_seg/images` và `masks`. CSV cũ ở `data/splits/*.csv` được giữ nguyên vì đã gắn hash với checkpoint cũ.

Split mới đã tạo tại `data/splits/grouped_v2`: 700 train, 150 validation, 150 test, 865 nhóm. Script sử dụng SHA256 file, hash pixel đã giải mã và pHash 63 bit; gộp bảo thủ mọi cặp Hamming <=8 và các thành phần liên thông. Nhóm không được chia qua các tập. Đường dẫn CSV xuất POSIX và dataloader đọc được cả dấu Windows/POSIX.

```bash
python -m src.split_dataset --output data/splits/grouped_v2 --validate-only
# Tạo một phiên bản mới, không ghi đè phiên bản đã có:
python -m src.split_dataset --output data/splits/grouped_v3
# Khi có metadata nguồn:
python -m src.split_dataset --output data/splits/patient_v1 --metadata data/patient_groups.csv
```

Metadata cần đúng một hàng cho mỗi ảnh, cột `image` và ít nhất một trong `patient_id`, `video_id`, `group_id`. Những ảnh có cùng định danh được gộp cùng nhóm, bao gồm quan hệ bắc cầu giữa các loại nhóm. Kiểm tra chất lượng metadata trước khi sử dụng.

**Giới hạn:** perceptual hash không tìm hết các frame liên quan và không xác nhận bệnh nhân/video. Manifest hiện tại ghi `not_verified_metadata_unavailable`; không được đổi thành “patient-independent” khi chưa có bằng chứng. Các cặp pHash <=8 là nhóm bảo thủ để ngăn ứng viên tương tự qua tập, không phải kết luận mọi cặp là cùng ảnh/bệnh nhân.

Manifest ràng buộc hash của CSV, groups và nội dung từng ảnh/mask. Trainer kiểm tra trước/sau train; evaluator kiểm tra trước/sau test. Dữ liệu hoặc nguồn training thay đổi thì chạy dừng và yêu cầu phiên bản/thí nghiệm mới.

## Huấn luyện đầy đủ trên CPU

Cấu hình chính: `configs/experiment_cpu.json`, cả hai kiến trúc, seed 42/43/44, tối đa 50 epoch, batch 4, AdamW 1e-4, BCE+Dice, early stopping 10, ReduceLROnPlateau patience 3/factor 0,5. Cả 6 lượt dùng một encoder ImageNet đã đóng băng theo SHA256.

```bash
python -m src.prepare_encoder
python -u -m src.run_experiments --config configs/experiment_cpu.json
```

Chạy lại cùng lệnh để tiếp tục thí nghiệm bị gián đoạn. Chỉ một runner được giữ file lock cho mỗi thí nghiệm. Trainer khôi phục model/optimizer/scheduler/RNG/shuffle generator và kiểm tra config, source, split cùng môi trường. Lượt đã hoàn tất được xác minh và bỏ qua; không train thêm sau early stopping. Không tự tắt máy.

Trạng thái: `results/experiments/grouped_v2_cpu/status.json`. Log riêng mỗi lượt ở cùng thư mục. Mỗi run có `config.json`, `environment.json`, `history.csv`, `last_checkpoint.pth`, `best_weights.pth`, `decision.json`, `run_complete.json`. Không ghi đè checkpoint E3/E4 cũ.

Augmentation chỉ áp dụng train: lật ngang/dọc và xoay 0/90/180/270 độ đồng thời ảnh/mask. Transform được xác định bởi `(seed, epoch, sample index)`, nên hai kiến trúc cùng seed thấy cùng shuffle và transform, đồng thời resume tái tạo đúng. Validation/test không augmentation.

Validation và test đều threshold tại độ phân giải mạng rồi đưa mask về kích thước gốc bằng nearest-neighbor để tính Dice/IoU macro theo ảnh; empty/empty=1. Loss validation vẫn ở kích thước mạng và không được diễn giải như metric kích thước gốc.

Checkpoint được chọn bằng validation Dice ở threshold 0,5. Sau đó mới tìm threshold trên grid định trước 0,30–0,70, bước 0,05, chỉ dùng validation; hòa điểm chọn gần 0,5 nhất. Cả 6 run phải hoàn tất, kiểm chứng cấu hình đồng nhất và đóng băng hash trước khi bất cứ run nào được tính test. Không chọn seed/epoch/threshold bằng test.

Kết quả cuối:

- `results/experiments/grouped_v2_cpu/summary.json`: mean và SD giữa seed, paired bootstrap CI95% theo seed và nhóm ảnh, kiểm định sign-flip cặp giữa seed.
- `test_samples.csv`: từng ảnh, seed, nhóm và cả hai protocol threshold.
- `frozen_runs.json`: hash toàn bộ quyết định trước test.
- `deployment.json`: hai checkpoint ở seed 42, được chọn trước theo cấu hình, không chọn seed tốt nhất trên test.

Endpoint chính được định trước là Dice test với threshold cố định 0,5. IoU và threshold chọn trên validation là phân tích phụ. Với 3 seed, kiểm định có độ phân giải thấp; không diễn giải không có ý nghĩa thống kê thành hai mô hình tương đương. Bootstrap không khắc phục thiếu metadata bệnh nhân/video. Những vấn đề nguồn dữ liệu còn thiếu được giữ trong báo cáo.

Chạy một lượt riêng:

```bash
python -m src.train --model unet --seed 42 --split-dir data/splits/grouped_v2 --run-name unet_manual_v2 --device cpu
python -m src.train --model unet --seed 42 --split-dir data/splits/grouped_v2 --run-name unet_manual_v2 --device cpu --resume
```

Khi resume cần giữ nguyên các đối số đã dùng. `src/train_polyp_thesis.py` chỉ là entry point tương thích gọi cùng implementation. Checkpoint v1 không được resume bằng protocol v2; `complete_training.py` và `shutdown_after_training.py` là công cụ giám sát thí nghiệm cũ, không dùng cho study mới.

## Kiểm thử và API

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/frontend_flow.test.cjs
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8082
```

Unit test dùng dữ liệu tổng hợp, không cần dataset thật, checkpoint thật hoặc server. Cần cài dependencies để chạy đầy đủ; những nhóm thiếu dependency báo skip rõ ràng. Test resume so sánh từng tensor, metric, shuffle và scheduler với run không gián đoạn.

Integration test là opt-in và chỉ skip nhóm integration khi thiếu tài nguyên:

```bash
# Linux
POLYP_RUN_INTEGRATION=1 POLYP_TEST_URL=http://127.0.0.1:8082 python -m backend.test_web_integration -v
# PowerShell
$env:POLYP_RUN_INTEGRATION='1'
$env:POLYP_TEST_URL='http://127.0.0.1:8082'
python -m backend.test_web_integration -v
```

`/api/study` hiển thị trạng thái và summary của study mới. `/api/evaluation` tiếp tục đánh giá checkpoint/split cũ, kèm cảnh báo ảnh gần trùng; đó không phải kết quả study mới. Giữ riêng hai nguồn kết quả để tránh đổi split nhưng vẫn dùng trọng số đã học từ các ảnh mới chuyển sang test.

CLI tự đọc kiến trúc, kích thước ảnh và threshold từ config/decision thực tế:

```bash
python -m src.predict --image test_images/polyp_test.jpg
python -m src.predict --image test_images/polyp_test.jpg --weights results/unetplusplus_grouped_v2_cpu_seed42/best_weights.pth --output results/new_prediction
```

## Cơ sở kỹ thuật

- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html): deterministic được bật và ghi môi trường; không bảo đảm bitwise giữa mọi phiên bản/hệ điều hành/GPU.
- [ReduceLROnPlateau](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html): scheduler theo validation và có state được checkpoint.
- [SMP architectures](https://smp.readthedocs.io/en/latest/models.html): hai kiến trúc cùng encoder ResNet34.

Để study tiếp tục chạy độc lập với cửa sổ lệnh/chat, dùng launcher thay cho lệnh trực tiếp:

```bash
python -m src.launch_experiments --config configs/experiment_cpu.json
```

Launcher không ngăn máy ngủ/tắt nguồn. Nếu bị gián đoạn, chạy lại cùng lệnh để resume từ checkpoint cuối. Không sửa source/config/split của một study đang chạy; tạo phiên bản mới khi thay đổi phương pháp.

Nếu chạy launcher trong khi study đã hoạt động, lệnh trả `status: already_running` và không tạo worker trùng. Launcher kiểm tra khóa trước khi đọc/ghi byte trên Windows. File `runner.lock` còn tồn tại sau khi tắt máy là bình thường; khóa hệ điều hành được nhả khi tiến trình kết thúc, vì vậy không cần xóa file này để resume.

## Theo dõi epoch trực tiếp

```powershell
.\.venv\Scripts\python.exe -u -m src.monitor_training
```

Màn hình tự làm mới mỗi 5 giây, hiển thị các lượt trong study, epoch đã lưu, train/validation loss, Dice/IoU, learning rate và thời gian epoch. Tự chuyển sang lượt tiếp theo. Chỉ số mới được ghi sau khi mỗi epoch hoàn tất; chưa có tiến độ từng batch. `Ctrl+C` chỉ dừng màn hình theo dõi, không dừng trainer nền.

Dùng `--once` để xem một lần, `--tail 12` để hiển thị 12 epoch gần nhất hoặc `--interval 10` để thay đổi chu kỳ làm mới. Nếu trainer chạy bằng Python Windows, dùng cùng Python Windows để kiểm tra PID; khi đọc từ Linux/WSL, monitor báo trạng thái PID chưa xác minh thay vì nhầm rằng trainer đã dừng.
