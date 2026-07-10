# KẾ HOẠCH TRIỂN KHAI TUYẾN MODEL SIZING + NỀN TẢNG CHO TUYẾN TIẾP THEO

> **QUYẾT ĐỊNH 2026-07-09 (sau khi lập kế hoạch): HOÃN toàn bộ tuyến sizing (GĐ1-2).**
> Lý do: production serving đang xây quanh mô hình 1 lệnh = 1 mua + 1 bán; pyramid đòi multi-lot
> (ledger schema, dashboard, margin ledger, thêm loại lệnh ADD cho trader) — chi phí kiến trúc lớn
> trong khi edge chỉ đáng kể khi dùng margin. Ưu tiên mới: **chất lượng tín hiệu** (GĐ3 - Tuyến A
> lên làm mũi chính). Kết quả pyramid đã đóng hồ sơ tái lập được: templates 2666-2681 trên
> leaderboard, `week0/WEEK0_RESULTS.md` (§6 audit), `week0/portfolio_sim_results.md` — khi hệ thống
> sẵn sàng quản lý nhiều điểm mua/bán sẽ kích hoạt lại, không cần nghiên cứu lại.
> Phần GIỮ LẠI từ GĐ0 (không cần code, áp ngay cho mọi tuyến): checklist promote + vòng công tố
> (mục 0.4). Các mục 0.1-0.3 (fill convention, cột capnorm, tool portfolio sim) chỉ cần khi tái
> kích hoạt tuyến sizing.

Ngày lập: 2026-07-09. Bối cảnh: quyết định adopt pyramid + margin 50%@15%/năm (portfolio sim đã verify: CAGR 54,3% vs 50,0%, MDD −18,5% vs −13,8%, lãi vay 0,69% lợi nhuận — `week0/portfolio_sim_results.md`). Kế hoạch này chuẩn hóa hệ thống để (a) đưa pyramid vào production sạch sẽ, (b) mọi tuyến model sau không bị thước đo đánh lừa như vụ "+35% composite".

## Nguyên tắc xuyên suốt (rút từ audit phiên này)

1. **Knob mới luôn gated default-off** + golden parity test: config all-off phải tái lập champion bit-identical (tiền lệ volhold, §11a stock-serving/CLAUDE.md).
2. **Đo lường trước, model sau**: không thí nghiệm sizing nào được chấm khi khung capital-aware chưa có mặt.
3. **Một thí nghiệm = một clone template**, append-only; MỌI run đăng ký leaderboard kể cả kết quả xấu — không cherry-pick.
4. **Vòng công tố bắt buộc**: mọi "breakthrough" phải qua adversarial fairness audit (như tmp_gap_fairness.py) trước khi báo cáo. Quy trình hóa, không tùy hứng.
5. **Không che vấn đề**: mục Caveats bắt buộc trong mọi report; scripts + CSV tái lập đặt cạnh report; kết quả ngược kỳ vọng phải ghi rõ, không "làm tròn".

## GIAI ĐOẠN 0 — Vá nền đo lường (làm TRƯỚC tất cả, ~1-2 ngày)

### 0.1 Engine (`stock_ml/src/backtest/engine.py`)
- **Fill convention của add**: thêm knob `pyramid_add_fill: str = "close_next"` (giữ `"close_same"` để tái lập số cũ). Default = convention trung thực, khớp `entry_bar_fill_type` của chính engine. (Đã đo: −2,98u, 12/524 add bất khả thi ở close_same.)
- **`pyramid_add_skip_washout: bool = True`**: không add khi market-drop/washout gate đang active — chống bơm margin đúng lúc thị trường sập (42,4% big loss nằm trong 3 tháng sập index).
- Unit tests mới: `test_pyramid_fill_convention`, `test_pyramid_washout_skip`, `test_pyramid_off_parity` (pyramid_add_units=0 → trades bit-identical golden).

### 0.2 Scoring / leaderboard (`stock_ml/src/evaluation/scoring.py`, bảng `leaderboard_runs`)
- **KHÔNG đổi composite hiện hành** (giữ so sánh lịch sử ~2.600 template).
- Thêm cột metrics mới (nullable, run cũ để NULL — không viết lại lịch sử): `unit_days`, `peak_units`, `pnl_per_unit_day`, `composite_capnorm` (composite tính trên pnl đã chia units-ratio). Tính trong calc_metrics khi trades có cột weight.
- Quy tắc đọc số: model có sizing chỉ được so bằng `composite_capnorm` + PnL/MDD + portfolio sim; composite thô chỉ để tham chiếu.

### 0.3 Portfolio sim thành công cụ chuẩn
- Promote `analysis/serving_blindspot/week0/portfolio_sim.py` → `stock_ml/tools/portfolio_sim.py`: CLI nhận trades CSV / run_id, tham số `--slots K --margin-cap 0.5 --margin-rate 0.15`; xuất bảng + daily NAV CSV.
- Sửa lookahead nhẹ đã biết (MTM interpolation về exit anchor): default = ratio đóng băng tại entry (hold0, verifier đã đo lệch ≤0,13pt DD); mode cũ sau flag.
- Unit tests: lãi margin khớp ledger recompute (chuẩn 0.00e+00 như verifier), skip logic đơn điệu theo K, no-lookahead.

### 0.4 Quy trình promote mới (checklist, thêm vào docs research + stock-serving/CLAUDE.md §11 như trục parity thứ 6: CAPITAL/SIZING frame)
Ứng viên chỉ được promote khi đủ: composite + composite_capnorm + portfolio sim (%NAV MDD, CAGR) + guardrail core-edge (BLINDSPOT_REPORT.md Phần 3) + multi-seed 42/7/99/555 + seed out-of-batch (123) + fairness audit + leakage-auditor PASS + bảng metric vận hành (limit sống/ngày, fill-rate, max loss/lệnh).

## GIAI ĐOẠN 1 — Tái chấm pyramid dưới thước đo sạch (~1 ngày máy)
1. Re-sweep `pyramid_add_units {0.5,1.0} × min_ret {0.02,0.03,0.04}` với `close_next` + `skip_washout` (clone 2646, pattern deploy_wavestruct.py).
2. Multi-seed + seed-123 cho top-2; portfolio sim margin 50%@15% K∈{25,30}.
3. Kỳ vọng: kết quả ≈ tuần-0 (test ±1 bar đã cho thấy robust, bar+4 còn tốt hơn). **Nếu lệch mạnh = có vấn đề chưa biết → DỪNG, điều tra, không ép số.**
4. Chốt config production + báo cáo bằng số honest (CAGR/%NAV MDD, không headline composite thô).

## GIAI ĐOẠN 2 — Đưa vào serving paper-trading (~2-4 ngày, sau GĐ1)

### 2.1 Wheel & bundle
- Rebuild wheel (engine đổi) → verify **no-drift bundle active** (knob default-off nên champion bit-identical — quy trình §11a). Cập nhật `dist/` + `requirements.txt`.
- Export bundle mới `export_bundle.py --replicate-last-fold`, pin universe + `--last-test-year` = ref.
- Manifest ghi thêm: pyramid knobs, margin params (cap 0.5, rate 0.15), feature_scope — deploy sau tự biết ép parity.

### 2.2 Serving (stock-serving)
- `serving/report.py` + worker: khối **ADD** — mỗi vị thế mở, đến bar thứ 3 sau fill nếu ret ≥ min_ret → dòng "ADD tại ATC hôm nay" (deterministic từ store, không cần model mới). View `pending_adds()` cạnh pending_exits. LƯU Ý: bỏ add nếu washout gate đang active (khớp engine).
- Ledger: thêm kind `add` (cột `kind` default 'signal', PK mở rộng; migration append-only, KHÔNG đụng rows cũ).
- **Margin ledger** cho track record trung thực: bảng draw/repay/interest cập nhật trong daily job — chi phí vay hiện rõ trên dashboard, không ẩn.
- Dashboard: cột units/vị thế, panel margin (dư nợ hiện tại/đỉnh/lãi lũy kế), số add chờ hôm nay.
- Tests: `test_report_add_rows`, `test_margin_ledger_interest`, giữ test_signal_log bất biến.

### 2.3 Shadow run 4-6 tuần
Bundle mới chạy song song champion (KEEP_BUNDLES hỗ trợ sẵn). Tiêu chí go-live: khớp ≥95% lệnh + add đúng lịch, drawdown thực trong kỳ nằm trong kỳ vọng sim, margin đỉnh < 40% NAV.

## GIAI ĐOẠN 3 — Tuyến A: kênh vào lệnh chân sóng (mũi phá trần thật, song song GĐ2)
1. **Chốt thiết kế trước khi code**: B-channel chỉ kích hoạt khi kênh A mù (có tín hiệu nhưng dưới MA20, hoặc vừa exit mà cấu trúc còn nguyên); vào at-market close_next CÓ xác nhận nến; stop = structural swing-low (`structural_stop_lookback` đã build); cap slot/ngày; giữ washout gate cho cả entry B.
2. Engine: nhóm knob `entry_bchannel_*` default-off + golden parity + unit tests; wheel rebuild đúng quy trình.
3. Research trên harness với khung đo mới; mục tiêu: ăn vào 7.582 tín hiệu không khớp (cận trên +768u/21bar) mà không rơi guardrail core-edge.
4. Vòng công tố bắt buộc trước khi báo kết quả.

## Kiểm soát lỗi & kỷ luật git
- Nhánh riêng cho từng giai đoạn (từ `refactor/phase-0.2-golden`); commit nhỏ; KHÔNG gộp engine-change với experiment-result cùng commit.
- Definition of done mỗi bước: tests pass + golden parity + run đăng ký leaderboard + report có Caveats.
- Thứ tự bắt buộc: GĐ0 → GĐ1 → (GĐ2 ∥ GĐ3). Không nhảy cóc: mọi số chấm trước khi GĐ0 xong đều coi là tạm.
