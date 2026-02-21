# 重庆高考录取预测系统（Web）

## 功能
- 分数/位次换算（历史类、物理类）
- 艺术类预测（按普高艺术综合分规则计算并进行录取概率分析）
- 院校专业录取概率预测（冲 / 稳 / 保）
- 智能推荐（支持院校、专业、地域关键词筛选）
- 概率悬停说明（鼠标放在概率列可查看“高把握/冲刺项”等解释）

## 数据文件
以下文件需与 `app.py` 在同一目录：
- `hist.xlsx`
- `phys.xlsx`
- `艺术类.xlsx` 或 `艺术批.xlsx`
- `历史类含加分一分段表.xlsx`
- `物理类含加分一分段表.xlsx`

联考换算功能（可选）支持多场考试切换：
- 在项目目录创建 `mock_exams` 文件夹
- 每场联考按科类分别放文件，建议命名：
  - `历史_一诊.xlsx`
  - `历史_二诊.xlsx`
  - `物理_一诊.xlsx`
  - `物理_二诊.xlsx`
- 文件需包含列：`分数段`、`累计人数`
- 若使用旧命名（如 `hist_mock.xlsx` / `phys_mock.xlsx`）也会兼容识别

## 启动
1. 安装 Python 3.10+  
2. 在项目目录执行：
   ```bash
   python -m pip install -r requirements.txt
   python app.py
   ```
3. 浏览器访问：`http://127.0.0.1:5000`

Windows 可直接双击 `web.bat` 启动。

艺术类综合分规则：`(文化素质测试和技术科目测试总成绩 / 750 * 300 * 50%) + 专业统考成绩 * 50%`，四舍五入保留两位小数。

联考换算逻辑：先由联考成绩映射联考位次，再按位次百分位映射到高考一分一段，得到预测高考分并参与志愿预测。
页面入口：
- 高考分预测页：`/`
- 联考换算页：`/mock-convert`
- About 页：`/about`（内容可在 `templates/_about_content.html` 自由编辑）

## Railway 部署（推荐测试环境）
1. 将项目推送到 GitHub（需包含 `hist.xlsx`、`phys.xlsx`、一分一段表文件）。
2. 在 Railway 新建项目，选择 `Deploy from GitHub repo`。
3. 等待构建完成后，确认服务启动命令为 `Procfile` 中的 gunicorn 命令。
4. 在 Railway 服务设置中配置：
   - Healthcheck Path: `/health`
   - Environment Variables: `RELOAD_TOKEN`（可选，保护 `/reload` 接口）
   - 生成公网域名（`*.up.railway.app`）
5. 浏览器访问 Railway 域名验证功能。

## Cloudflare 免费接入（域名与加速）
1. 将你的域名接入 Cloudflare（使用 Cloudflare 分配的 NS）。
2. 在 Cloudflare DNS 新增：
   - Type: `CNAME`
   - Name: `gaokao`（示例）
   - Target: Railway 提供的公网域名（`xxx.up.railway.app`）
3. 先关闭代理（灰云）完成 Railway 域名验证；验证通过后可改为橙云代理。
4. SSL/TLS 设置为 `Full (strict)`（前提是源站证书有效）。
5. 访问 `https://gaokao.你的域名` 完成公网访问。
