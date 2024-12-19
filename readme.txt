1. Cài đặt các gói cần thiết
    pip install -r requirements.txt

2. Chạy script crawl dữ liệu cho từng đội bóng
    python run_cmd <index>
với <index> là thứ tự của từng đội bóng. Chạy lần lượt với <index> từ 0 tới 11. Không thể chạy cùng lúc vì bị chặn spam request từ fbref.com.

3. Thay đổi các thông số thử nghiệm trong config.yml
    model_type: Loại mô hình sử dụng (`mlp` cho mạng neuron, `sr` cho hồi quy logistic đa thức)
    data_files: Chuỗi đặc tả nơi lưu trữ file dữ liệu (ví dụ: `data/*.jsonl`)
    save_steps: Số bước lưu checkpoint
    model_args:
        hidden_size: Kích thước vector biểu diễn
        num_teams: Số đội
        num_venues: Số loại sân
        num_results: Số kết quả

    trainer:
        accelerator: Loại thiết bị sử dụng cho tính toán (`cuda`, `mps`, `cpu`, ...)
        strategy: Chiến thuật sử dụng phần cứng
        devices: Số lượng thiết bị sử dụng
        max_steps: Số lượng bước huấn luyện cần chạy
        accumulate_grad_batches: Số bước tích luỹ đạo hàm
        precision: Cấu hình số bit biểu diễn số thực (`16`, `16-mixed`, `32`, ...)

4. Chạy script huấn luyện
    python main.py

5. Chạy script đánh giá
    python eval.py
* Lưu ý: Thêm trường checkpoint vào config.yml lưu đường dẫn tới checkpoint cần đánh giá.