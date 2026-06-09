# Bug Report：页面显示裂图（破损图片）

## 现象

Cloudflare Pages 部署后，浏览器访问页面时出现一张加载不出来的破损图片。

## 已尝试的修复

| 步骤 | 操作 | 结果 |
|------|------|------|
| 1 | 删除博客文章的 `heroImage` frontmatter | 无效 |
| 2 | 删除 `BaseHead.astro` 的 `FallbackImage` 默认图 | 无效 |
| 3 | 将 `og:image` 和 `twitter:image` 改为条件渲染 | 无效 |

## 排查结果

```bash
# 本地构建产物确认
grep -r '<img' dist/ --include="*.html"
# 输出：空（没有任何 <img> 标签）

grep -r 'og:image' dist/ --include="*.html"
# 输出：空（没有任何 og:image meta 标签）
```

## 关键信息

- **本地构建**：`npm run build` 通过，`dist/` 产物中不含任何 `<img>` 标签
- **本地 dev server**：`npm run dev` → `http://localhost:4321` 正常显示
- **部署平台**：Cloudflare Pages（通过 GitHub 连接，push 自动构建）
- **站点地址**：https://my-blog.15614533.workers.dev
- **GitHub 仓库**：https://github.com/MICHAELWAY617/my-blog
- **代码分支**：master（最新 commit: 21a6cd3）

## 可能的根因

1. **Cloudflare 缓存了旧版本** — 即使新 push 构建成功，CDN 边缘节点可能还在返回旧的 HTML
2. **Cloudflare Workers 注入** — 部署被识别为 Workers（有 KV Namespace、Images binding），可能有额外的资源注入逻辑
3. **浏览器缓存** — 用户浏览器可能缓存了旧页面
4. **favicon 加载失败** — `favicon.ico` / `favicon.svg` 在某些浏览器上显示为裂图
5. **Cloudflare 构建未更新** — 可能构建失败了，但页面回退到了上一个成功构建（含 heroImage）

## 建议的排查步骤

1. 打开 Chrome DevTools → Network 标签 → 刷新页面，找到那个加载失败的请求，看它的 URL 是什么
2. 在 Cloudflare Dashboard → Workers & Pages → my-blog → Deployments，确认最新部署的状态和时间
3. 用无痕窗口打开 https://my-blog.15614533.workers.dev ，看是否仍有裂图
4. 运行 `npx wrangler pages deployment list --project-name=my-blog` 查看部署历史
5. 如果确认最新代码没问题，在 Cloudflare Pages 后台手动 "Retry deploy" 或清除缓存

## 项目结构

```
my-blog/
├── astro.config.mjs      # site: https://my-blog.15614533.workers.dev
├── src/
│   ├── components/
│   │   ├── BaseHead.astro    # 已移除 FallbackImage
│   │   ├── Footer.astro
│   │   ├── FormattedDate.astro
│   │   ├── Header.astro
│   │   └── HeaderLink.astro
│   ├── content/
│   │   ├── config.ts
│   │   └── blog/
│   │       └── cnn-from-scratch.md  # 已移除 heroImage
│   ├── layouts/
│   │   └── BlogPost.astro
│   ├── pages/
│   │   ├── index.astro       # 首页（文章列表）
│   │   ├── about.astro
│   │   ├── blog/
│   │   │   ├── index.astro   # 博客列表
│   │   │   └── [...slug].astro
│   │   └── rss.xml.js
│   └── styles/
│       └── global.css
└── public/
    ├── favicon.ico
    └── favicon.svg
```
