# WayLog 维护日志

## 2026-06-09：部署访问故障排查

### 问题现象

网站部署到 Vercel 后，浏览器访问 `*.vercel.app` 地址**无法看到任何内容**。

### 排查过程

| 步骤 | 操作 | 结果 |
|------|------|------|
| 1 | `npm run build` 本地构建 | ✅ 成功，4 个页面正常生成 |
| 2 | 查看 `dist/index.html` 产物 | ✅ HTML 结构完整，中文内容正常 |
| 3 | `npm run dev` 启动本地 server | ✅ `http://localhost:4321` 一切正常 |
| 4 | `curl https://my-blog-xxx.vercel.app` | ❌ exit code 28，连接超时 |
| 5 | `npx vercel ls` | ✅ 部署状态 Ready，Vercel 平台侧正常 |

### 根因分析

从当前网络环境无法连接到 `*.vercel.app` 域名，**Vercel 默认域名在某些网络环境下不可达**。

可能原因：
- DNS 解析失败或被污染
- 防火墙 / 网络策略拦截
- 区域性网络限制

### 当前状态

- GitHub 仓库：https://github.com/MICHAELWAY617/my-blog ✅ 正常
- Vercel 平台构建：✅ 正常（日志显示 Ready）
- 本地开发：`npm run dev` → `http://localhost:4321` ✅ 正常
- 公网访问：❌ 无法连接

### 解决方案选项

#### 方案 A：Vercel 自定义域名（推荐）

1. 准备一个域名（可在 Cloudflare / Namecheap 等购买）
2. 在 Vercel 项目后台 `Settings → Domains` 添加域名
3. 在域名 DNS 添加 CNAME 记录指向 Vercel
4. 非 `*.vercel.app` 域名，通常不受限制

#### 方案 B：GitHub Pages

```bash
# 安装 @astrojs/github-pages 适配器
npx astro add github-pages
# 配置 astro.config.mjs 的 site 和 base
# 部署到 https://michaelway617.github.io/my-blog/
```

GitHub Pages 在国内基本可访问，但速度可能偏慢。

#### 方案 C：Cloudflare Pages

1. 注册 Cloudflare 账号
2. 连接 GitHub 仓库
3. 框架预设选 Astro，构建命令 `npm run build`，输出目录 `dist`
4. 自动获得 `*.pages.dev` 域名，通常比 Vercel 更快

#### 方案 D：国内平台

- 腾讯云静态网站托管
- 阿里云 OSS + CDN
- 需备案

### 暂时方案

在解决公网访问前，可以：

```powershell
cd D:\Projects\my-blog
npm run dev
```

浏览器打开 `http://localhost:4321` 本地预览，开发体验完全一致。

---

## 部署历史

| 时间 | Commit | 内容 | 状态 |
|------|--------|------|------|
| 20:08 | `420b204` | 首页改版：显示文章列表 | ✅ |
| 19:59 | `d9ee894` | 上传 CNN 源码 + 第一篇技术日志 | ✅ |
| 19:53 | `2b6766f` | 全站中文化 | ✅ |
| 19:47 | `ac63083` | 修复 about 页面 pubDate | ✅ |
| 19:23 | `5f0dfcd` | 自定义站点配置、清空示例 | ✅ |
| 19:19 | `bbfcf47` | 初始提交：Astro blog 模板 | ✅ |
