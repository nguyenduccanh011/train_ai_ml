# Model Library - User Guide

**Status**: ✅ Active (DB-First)
**Last Updated**: 2026-06-01

Model Library là kho tái sử dụng 3 loại thành phần cấu thành một Strategy Template:

| Catalog | Mô tả | Tạo/sửa được? |
|---------|-------|----------------|
| **Feature Sets** | Tập feature đầu vào (vd `leading_v2` = 36 cột) | ❌ Chỉ đọc (định nghĩa trong code) |
| **Targets** | Biến mục tiêu cho ML (vd `trend_regime`) | ✅ Tạo/sửa qua API |
| **Components** | Model/rule cho slot entry/exit/regime/size | ✅ Tạo/sửa/xóa qua UI + API |

Sau khi tạo component, bạn lắp chúng vào template — xem [Template Submission Guide](TEMPLATE_SUBMISSION_GUIDE.md) và [Component Slots Architecture](guides/COMPONENT_SLOTS_ARCHITECTURE.md).

---

## Truy cập

### UI
```
http://localhost:8000/dashboard/model-library.html
```
- Liệt kê toàn bộ component (entry/exit/regime/size)
- Nút **"+ New Component"** để tạo mới
- Sửa params / xóa (soft-delete) từng component

### API base
```
http://localhost:8000/api/v1/model-library
```
Swagger: `http://localhost:8000/api/docs`

### Khởi tạo dữ liệu mặc định
Catalog được nạp khi import template từ YAML:
```bash
python stock_ml/scripts/import_yaml_templates.py --commit
```
Mặc định sau khi seed: feature set `leading_v2`, target `trend_regime`, 2 rule component mẫu (`macd_ma20_entry`, `macd_ma20_exit`).

---

## 1. Feature Sets (chỉ đọc)

Feature set được **định nghĩa trong code** (vd `stock_ml/src/features/leading_v2.py`) và đăng ký vào catalog qua `import_yaml_templates.py`. Không tạo/sửa được từ UI/API — chỉ chọn khi build template.

### List feature sets
**GET** `/feature-sets`
```json
[
  {
    "id": 1,
    "name": "leading_v2",
    "columnCount": 36,
    "columns": ["macd_hist", "sma_20_ratio", ...],
    "description": "..."
  }
]
```

### Get chi tiết
**GET** `/feature-sets/{feature_set_id}`

> Muốn thêm feature set mới (vd `leading_v3`): viết module feature trong code rồi thêm vào `init_feature_sets()` của `import_yaml_templates.py`. Đây là việc của developer, không phải thao tác UI.

---

## 2. Targets

Target = biến mục tiêu ML dự đoán (phân loại regime, forward return…).

### List targets
**GET** `/targets?type={type}` (param `type` tùy chọn)
```json
[
  {
    "id": 1,
    "name": "trend_regime",
    "type": "trend_regime",
    "params": { ... },
    "outputDtype": "int32",
    "description": "..."
  }
]
```

### Tạo target
**POST** `/targets`

Bắt buộc: `name`, `type`, `params`, `outputDtype`.
```json
{
  "name": "fwd_return_5d",
  "type": "forward_return",
  "params": { "horizon": 5 },
  "outputDtype": "float32",
  "description": "5-day forward return regression target"
}
```
**409** nếu `name` đã tồn tại.

### Cập nhật target
**PUT** `/targets/{target_id}` — chỉ sửa `params` và/hoặc `description`.

### Get
**GET** `/targets/{target_id}`

---

## 3. Components

Component là khối dùng được cho 1 slot: **entry / exit / regime / size**.

### Trường dữ liệu
| Field | Bắt buộc | Giá trị |
|-------|----------|---------|
| `name` | ✅ | Định danh duy nhất |
| `role` | ✅ | `entry` \| `exit` \| `regime` \| `size` |
| `algorithm` | ✅ | `lightgbm` \| `xgboost` \| `random_forest` \| `mlp` \| `rule` |
| `params` | — | Object cấu hình (mặc định `{}`) |
| `description` | — | Mô tả tự do |
| `componentType` | (auto) | `ml` \| `rule` — phân loại, **xem Lưu ý bên dưới** |

### List components
**GET** `/components?role={role}&algorithm={algo}&component_type={ml|rule}`
Tất cả filter đều tùy chọn; không truyền thì trả về toàn bộ.
```json
[
  {
    "id": 1,
    "name": "macd_ma20_entry",
    "role": "entry",
    "algorithm": "rule",
    "componentType": "rule",
    "params": { ... },
    "description": "...",
    "createdAt": "2026-05-30T..."
  }
]
```

### Tạo component
**POST** `/components` — bắt buộc `name`, `role`, `algorithm`. **409** nếu trùng `name`.

### Get / Update / Delete
- **GET** `/components/{id}`
- **PUT** `/components/{id}` — chỉ sửa `params` / `description`
- **DELETE** `/components/{id}` — soft-delete (`is_active=false`)

---

## 3a. Tạo Rule Component

`algorithm: "rule"`. Logic đặt trong `params`:

```json
{
  "name": "rsi_oversold_entry",
  "role": "entry",
  "algorithm": "rule",
  "params": {
    "conditions": [
      {"feature": "rsi_14", "op": "<", "value": 30},
      {"feature": "close_to_open", "op": ">", "value": 1.0}
    ],
    "logic": "AND",
    "score_feature": "rsi_14"
  },
  "description": "Entry khi RSI < 30 và nến xanh"
}
```

**Toán tử hỗ trợ:** `>` `<` `>=` `<=` `==` `!=`
**Logic:** `AND` (mọi điều kiện đúng) hoặc `OR` (ít nhất 1 đúng).
`score_feature` (tùy chọn): feature dùng để xếp hạng/độ mạnh tín hiệu.

Feature dùng trong `conditions` phải thuộc feature set đã chọn ở template (vd `leading_v2`). Tên feature phân biệt hoa thường. Danh sách feature đầy đủ: xem [Template Submission Guide](TEMPLATE_SUBMISSION_GUIDE.md#available-features-for-rule-conditions).

---

## 3b. Tạo ML Component

`algorithm` là một trong `lightgbm | xgboost | random_forest | mlp`. `params` là hyperparameter của model:

```json
{
  "name": "lgbm_entry_v1",
  "role": "entry",
  "algorithm": "lightgbm",
  "params": {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31
  },
  "description": "LightGBM entry, predict trend_regime"
}
```

> Mặc định DB **chưa** seed sẵn ML component nào — bạn tự tạo. ML component cần kết hợp với một **Target** ở bước build template.

---

## Trường `component_type`

`component_type` (`ml` | `rule`) là bộ phân loại dùng cho filter dual-slot. Giá trị được **suy ra tự động từ `algorithm`** khi tạo component:
- `algorithm="rule"` → `component_type="rule"`
- mọi algorithm khác (`lightgbm`, `xgboost`, `random_forest`, `mlp`) → `component_type="ml"`

Không cần (và không thể) gửi `component_type` trong body POST — nó luôn khớp với `algorithm`.

> **Component cũ:** Component tạo trước bản vá này có thể bị gán nhầm `component_type="ml"` dù `algorithm="rule"`. Backfill một lần nếu cần dùng filter `?component_type=rule`:
> ```sql
> UPDATE model_components SET component_type='rule' WHERE algorithm='rule';
> ```

---

## Quy trình điển hình

1. (Tùy chọn) Tạo **Target** mới — `POST /targets`.
2. Tạo **Component** entry/exit — UI model-library.html hoặc `POST /components`.
3. Ghi lại `id` các component vừa tạo.
4. Vào **template-builder.html**, chọn Feature Set + Target + (Entry/Exit) component → lưu template.
5. Submit template để chạy backtest — xem [Template Submission Guide](TEMPLATE_SUBMISSION_GUIDE.md).

---

## Troubleshooting

| Lỗi | Nguyên nhân / cách xử lý |
|-----|--------------------------|
| `409 Component already exists` | `name` trùng — đổi tên |
| `400 Missing required fields` | Thiếu `name`/`role`/`algorithm` |
| Rule component cũ không hiện ở filter `component_type=rule` | Component tạo trước bản vá — chạy backfill SQL ở mục "Trường component_type" |
| `Features not found` khi backtest | Feature trong rule không có trong feature set đã chọn (phân biệt hoa thường) |
| Component không xóa được | Đang được template tham chiếu — gỡ khỏi template trước |
