# Bài phản tư — Lab 22: DPO/ORPO Alignment

**Tên:** Võ Huy Hoàng
**Mã học viên:** 2A202602548
**Khoá:** A20-K4 · Track 3
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

Bản phân tích được soạn với hỗ trợ AI từ kết quả chạy thật, giữ nguyên các kết quả không cải thiện. Nguồn số liệu: `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`, `data/eval/judge_summary.json`, `data/eval/side_by_side.jsonl` và `colab/Lab22_DPO_T4_executed.ipynb`. Không tuyên bố đã chạy bonus hoặc chạy lại toàn bộ pipeline trong một môi trường sạch.

## 1. Cấu hình và dữ liệu

| Mục | Giá trị thực tế |
|---|---|
| GPU / VRAM | Tesla T4; log Unsloth báo 14.563 GiB khả dụng |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Độ dài tối đa / seed | 768 token / 42 |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| LoRA (adapter tinh chỉnh nhỏ) | r=16, alpha=32; 33.030.144 tham số huấn luyện |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, lọc Vietnamese; 800 train / 100 eval |
| Chia dữ liệu | Theo prompt; kiểm tra không trùng câu hỏi đã qua |
| Chosen dài hơn rejected | 65.875%, được notebook làm tròn thành 65.9% |
| Trung vị độ dài chosen / rejected | 94 / 86 token |
| DPO: beta / learning rate / epoch | 0.1 / 5e-6 / 1 |
| Reference (mô hình tham chiếu) | SFT đã gộp tại `/content/lab22/models/sft-merged`, log-prob tính trước |
| DPO loss | sigmoid |
| Giám khảo cuối | `Skywork/Skywork-Reward-V2-Llama-3.2-3B` |
| Chi phí | Không gọi API giám khảo; chi phí GPU không được ghi nhận trong artifact |

### NB0 — hàm loss và câu hỏi dịch chuyển xác suất

`my_dpo_loss` dùng `-logsigmoid(beta * ((pc-rc) - (pr-rr))).mean()`. Hàm khớp bản tham chiếu với loss 0.6981. Kiểm tra chính hàm này khi policy bằng reference cho loss 0.6931, bằng log 2.

Margin có thể tăng khi xác suất chosen giảm vì mục tiêu DPO phụ thuộc chênh lệch thay đổi log-xác suất của chosen và rejected so với reference. Trong ví dụ NB0, chosen có reward -3 và rejected -5 thì margin vẫn là +2. Chosen bị giảm xác suất nhưng rejected bị giảm mạnh hơn. Vì vậy chỉ nhìn loss giảm hoặc margin tăng chưa đủ để kết luận câu trả lời tốt hơn.

### NB1 — SFT

Training hoàn tất 125/125 bước. Loss được log ở bước 10 là 1.884150 và ở bước 120 là 1.283601; loss trung bình toàn lần chạy là 1.3602. Biểu đồ cho thấy xu hướng giảm có dao động. Adapter SFT và mô hình merged 16-bit đã được lưu trong Colab.

### NB2 — đọc ba cặp mẫu

Các cặp được lưu nguyên văn trong `data/eval/preference_samples.json` và in trong cell audit cuối phần bắt buộc của notebook.

1. **Tạo 10 yêu cầu thay đổi:** chosen đánh số đủ 1–10 và dùng định dạng Trước/Yêu cầu/Sau nhất quán hơn. Rejected cũng có 10 mục nhưng hai mục không đánh số; có các cụm dịch khó hiểu. Có lý do về định dạng để ưu tiên chosen, nhưng đây không phải bằng chứng rằng trả lời dài hơn luôn tốt hơn.
2. **Phân loại bài đăng thù địch bằng tiếng Tây Ban Nha:** chosen trả lời “Phản ứng: Thô bạo”, rejected trả lời “Phản ứng: Bạo lực”. Cả hai không dùng đúng nhãn “Hung hăng”/“Không hung hăng” mà prompt yêu cầu. Không có cơ sở mạnh để coi chosen là đáp án chuẩn. Mẫu này cho thấy nhãn preference có thể nhiễu và dữ liệu lọc Vietnamese vẫn chứa đầu vào ngôn ngữ khác.
3. **Hướng dẫn đặt hẹn đánh giá giọng hát:** hai câu trả lời đều đưa các bước tương tự và đều khẳng định việc đặt hẹn đã thành công, dù chưa có hành động đặt hẹn nào. Nhãn chosen không giải quyết lỗi này. Một đáp án tốt hơn phải phân biệt hướng dẫn người dùng với xác nhận thao tác thực tế.

Không lọc lại ba mẫu này sau khi xem kết quả; split và kết quả hiện tại được giữ nguyên để tránh thay đổi dữ liệu làm mất tính nhất quán của đánh giá.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Tiến trình huấn luyện | 100/100 bước, 1 epoch |
| Thời gian trên thanh training | 30 phút 57 giây; không gồm toàn bộ thời gian tính reference và các cell khác |
| VRAM cao nhất | Không có phép đo riêng được lưu cho NB3; không suy ra từ VRAM hiện tại |
| First logged loss | 0.6916665 |
| Final train loss | 0.6737927 |
| Reward chosen cuối train | 0.4122747 |
| Reward rejected cuối train | 0.3165085 |
| Reward gap cuối train | 0.0957663 |
| Reward chosen held-out | 0.4244377 |
| Reward rejected held-out | 0.3358700 |
| Reward gap held-out | 0.0885676 |
| Reward accuracy held-out | 0.71 |
| Chẩn đoán tự động | INTENDED |
| Độ dài trung bình held-out SFT → DPO | 628.88 → 639.84 ký tự |

## 3. Đọc đường reward

Ảnh: `screenshots/03-dpo-reward-curves.png`.

Implicit reward (điểm ưu tiên ngầm) được tính bằng beta nhân log-tỉ lệ xác suất giữa policy và reference. Trong lần chạy này, reward cuối của chosen và rejected đều dương trên cả train và held-out. Chosen đạt 0.4123 còn rejected đạt 0.3165 trên train, tạo gap 0.0958. Trên held-out, hai giá trị là 0.4244 và 0.3359, tạo gap 0.0886. Do đó margin dương vì chosen được tăng tương đối nhiều hơn rejected; không thể mô tả kết quả là rejected giảm. Chẩn đoán INTENDED của helper chỉ yêu cầu chosen dương và margin dương, nên nhãn này rộng hơn kịch bản lý tưởng chosen tăng/rejected giảm.

Gap held-out gần gap train và reward accuracy held-out là 71%, cho thấy tín hiệu học sở thích trên dữ liệu để riêng. Tuy nhiên, chỉ hai giá trị cuối không đủ để loại trừ overfitting (học quá sát dữ liệu huấn luyện); cần xem cả đường cong và kiểm tra bằng đánh giá đầu ra. NB4 không cho thấy ưu thế có ý nghĩa của DPO, nên reward accuracy 71% không được dùng thay cho win rate chất lượng câu trả lời.

Likelihood displacement (dịch chuyển xác suất) không phải chẩn đoán của lần chạy này vì chosen cuối vẫn dương. Về độ dài, log-prob của chuỗi là tổng các log-prob token, nên câu dài thường có tổng âm hơn. DPO tối ưu tỉ lệ tương đối so với reference, không đơn giản thưởng tổng log-prob lớn. Khi nhãn preference ưu tiên câu dài, mô hình vẫn có thể học thiên vị độ dài. SimPO chuẩn hoá log-prob theo số token; ORPO trong helper cũng dùng log-prob trung bình trong thành phần odds-ratio. Các cách này thay đổi ảnh hưởng độ dài nhưng không tự sửa nhãn nhiễu hay đảm bảo đầu ra tốt hơn.

## 4. So sánh SFT và SFT+DPO

Ảnh: `screenshots/04-side-by-side-table.png`.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate và CI 95% | Win rate độ dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 3 | 8 | 39 | 45.00% [38.00%, 51.00%] | 45.74% (47 cặp) | 45.45% |
| hữu ích | 4 | 1 | 1 | 2 | 50.00% [12.50%, 87.50%] | 50.00% (2 cặp) | 50.00% |
| an toàn | 4 | 0 | 0 | 4 | 50.00% [50.00%, 50.00%] | 50.00% (4 cặp) | Không xác định: toàn bộ hoà |

Win rate tính mỗi lần hoà là nửa điểm: `(3 + 0.5 * 39) / 50 = 0.45`. CI held-out chứa 0.5 nên chưa đủ bằng chứng DPO tốt hơn SFT. CI [0.5, 0.5] ở bốn câu safety là kết quả bootstrap trên toàn bộ quan sát hoà, không chứng minh hiệu năng safety tổng quát không có bất định.

**Giám khảo:** Qwen3-4B chỉ qua 8/12 sanity pairs (66.67%), dưới ngưỡng 80%, nên bị loại. Llama-3.2-3B qua 12/12 (100%) và là giám khảo duy nhất trong kết quả cuối. Qwen cho held-out win rate 57%, Llama cho 45%; không chọn giám khảo có điểm cao hơn để báo cáo. Mức đồng thuận giữa hai giám khảo trên 58 câu là 81.03%. Dữ liệu sanity nhỏ nên 100% không đảm bảo độ tin cậy trên mọi câu hỏi tiếng Việt. Hai reward model đều thuộc Skywork, còn có nguy cơ preference leakage (rò rỉ sở thích) với nguồn nhãn.

Độ dài held-out tăng khoảng 1.74%, từ 628.88 lên 639.84 ký tự. Win rate sau ghép độ dài gần bằng là 45.74%, gần win rate 45%. Spearman giữa score và độ dài của giám khảo Llama là -0.0232, gần 0; với mẫu này chưa thấy tương quan mạnh. Không suy rộng thành kết luận giám khảo hoàn toàn không thiên vị độ dài.

**Ví dụ hữu ích — h2:** cả SFT và DPO trả lời giống nhau, gợi ý gà nướng, bánh mì và phô mai dù prompt chỉ nêu gạo, trứng. Cả hai không bám ràng buộc nguyên liệu. DPO không sửa được lỗi instruction-following (làm theo yêu cầu) này. Với h1, DPO giải thích quicksort dài hơn nhưng không thể dùng độ dài làm bằng chứng tự động tốt hơn.

**Ví dụ an toàn — s2:** hai mô hình cho câu trả lời giống nhau, từ chối viết tin nhắn đe doạ và gợi ý trao đổi tôn trọng. Hành vi từ chối đã có ở SFT; không có bằng chứng DPO tăng an toàn ở ví dụ này hoặc trên bốn câu cố định.

Nhiều đầu ra chứa literal `<tool_call>` hoặc `</tool_call>` không phục vụ câu trả lời. h4 còn lặp ý và có nhận định không chính xác về Python/JavaScript. Đây là vấn đề chất lượng có thật; chưa xác định nguyên nhân gốc từ artifact nên không quy kết chắc chắn cho tokenizer, template hay DPO. Đánh giá production cần kiểm tra riêng các lỗi này trước khi triển khai.

## 5. Đánh đổi theo beta — bonus

Không chạy beta-sweep. Chưa có so sánh thực nghiệm giữa beta 0.05, 0.1 và 0.5; không điền số liệu giả.

## 6. Một quyết định quan trọng nhất

Quyết định được phân tích là giữ đánh giá held-out cùng ngưỡng sanity 80% của giám khảo, thay vì xem reward accuracy hoặc chọn giám khảo có win rate cao hơn làm bằng chứng hoàn thành alignment (căn chỉnh hành vi). Cấu hình đã chạy hai reward model lần lượt trên T4, kiểm tra khả năng phân biệt các cặp tiếng Việt đơn giản, rồi loại mô hình không đạt. Phương án thay thế là dùng riêng Qwen3 vì nó cho DPO win rate 57%, hoặc bỏ qua sanity check. Cách đó làm kết luận có vẻ tích cực hơn nhưng không đáng tin khi Qwen chỉ đúng 8/12 sanity pairs.

Kết quả của quy trình hiện tại là giữ giám khảo Llama, có 12/12 sanity pairs đúng, và báo win rate 45% với CI 38–51%. Điểm này không xác nhận ưu thế của DPO dù reward accuracy NB3 là 71%. Sự khác biệt cho thấy cần tách metric tối ưu trong training khỏi chất lượng câu trả lời cuối cùng. Năm mươi câu held-out, nhiều lần hoà và chỉ một giám khảo còn hợp lệ là hạn chế của thí nghiệm, không phải lý do để bỏ các kết quả bất lợi.

Nếu làm lại, sẽ mở rộng sanity pairs bằng các lỗi định dạng, tuân thủ ràng buộc và ví dụ tiếng Việt sát dữ liệu; tăng số câu held-out trong phạm vi GPU cho phép; kiểm tra nhãn preference nhiễu trước training. Cần xử lý nguyên nhân đầu ra chứa tool-call markers rồi đánh giá trên một split cố định. Giám khảo khác họ qua API có thể là đối chứng tương lai nhưng chưa được chạy và có thể phát sinh chi phí. Đồng thời nên bổ sung checkpoint định kỳ và sao lưu từng phần để mất phiên Colab không làm mất tiến độ. Các thay đổi này là đề xuất cho lần chạy sau, không được mô tả như thí nghiệm đã thực hiện.

## 7. Benchmark — bonus NB6

Không chạy.

## 8. Biến thể loss — bonus NB3b

Không chạy.

## 9. GRPO — bonus NB7

Không chạy.

## Lưu bằng chứng và tái lập

Notebook đã chạy: `colab/Lab22_DPO_T4_executed.ipynb`; giữ output NB0–NB4. Dữ liệu, ảnh và JSON lấy trực tiếp từ runtime. Bằng chứng và trọng số hai adapter đã sao lưu tại `MyDrive/Lab22/run-20261008-154050`. ZIP bằng chứng chỉ chứa config của merged model; trọng số merged SFT không nằm trong ZIP này. Không commit trọng số hoặc khoá API.

`adapters/dpo/adapter_config.json` giữ nguyên đường dẫn Colab `/content/lab22/models/sft-merged`; không sửa thành đường dẫn Windows để làm sai lệch nguồn chạy. Verifier gốc cần chạy tại `/content/lab22` với mã nguồn, REFLECTION và các artifact đầy đủ. Lần chạy này có output thực tế từng phần, nhưng chưa chạy lại toàn bộ pipeline từ môi trường sạch.

`make verify` đã chạy tại `/content/lab22` và kết thúc với mã 0. Log gốc nằm ở `submission/verify_colab.log`, metadata ở `data/eval/verification.json`, và ảnh xác nhận ở `submission/screenshots/core-verify-colab.jpg`. Verifier không bị sửa để bỏ qua kiểm tra. Repo GitHub đã public; việc nộp LMS chưa hoàn tất vì trang VLearn báo cần đăng nhập bằng tài khoản học viên đã được cấp.
